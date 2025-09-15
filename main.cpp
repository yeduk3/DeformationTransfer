#include "error.hpp"
#include "glm/ext/matrix_transform.hpp"
#include <YGLWindow.hpp>
#include <cmath>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <set>
#include <queue>
#include <algorithm>
YGLWindow* window;
#include <objreader.hpp>
ObjData srcOriginal, tarOriginal;
// ObjData cat, lion;
// ObjData horse, camel;
#include <program.hpp>
Program shader;
#include <camera.hpp>
#include <filesystem>
#include <chrono>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/Dense>

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::Triplet;
using Eigen::Matrix3d;
using Eigen::SparseLU;
using std::vector;
using Triangle = Eigen::Vector3<unsigned short>;
using TriangleMesh = vector<Triangle>;
using Vertices = vector<Vector3d>;


template <typename T, size_t dim>
struct Node {
    Node() : v(dim) {}
    int            id;
    std::vector<T> v;
};
template <typename T, size_t dim>
struct KDTree {
    struct KDTreeNode : Node<T, dim> {
        int         axis = -1;
        KDTreeNode* l    = nullptr;
        KDTreeNode* r    = nullptr;
        ~KDTreeNode() { delete l; delete r; }
    };
    
    KDTreeNode* root;
    std::vector<T> data;
    std::vector<size_t> dataIdx;
    KDTree() {}
    KDTree(const std::vector<T>& pointSet) : data(pointSet) {
        // printf("Build KDTree with size: %zu\n", data.size());
        dataIdx.resize(nodeCount());
        std::iota(dataIdx.begin(), dataIdx.end(), 0);
        
        build();
    }
    
    size_t nodeCount() { return data.size()/dim; }
    
    // Build----------------------------- Expected: O(N log N)
    
    void swapPoint(size_t a_idx, size_t b_idx) {
        if(a_idx == b_idx) return;
        T* a = &data[a_idx*dim];
        T* b = &data[b_idx*dim];
        for(int i = 0; i < dim; i++) std::swap(a[i], b[i]);
        std::swap(dataIdx[a_idx], dataIdx[b_idx]);
    }
    size_t partitionWithMedian(size_t begin, size_t end, const std::function<bool(size_t a_idx, size_t b_idx)>& comp) {
        if(end-begin < 1) return -1;
        size_t med = (begin+end)/2;
        
        auto left  = begin;
        auto right = end-1;
        while(left < right) {
            size_t pivot = left;
            for(size_t i = left; i < right; i++) {
                if(comp(i, right)) {
                    // if any element is smaller than end_index element, send it to begin_direction
                    swapPoint(i, pivot);
                    pivot++;
                }
            }
            swapPoint(pivot, right);
            // {low} (pivot=prevend) {high}
            if(pivot == med) break;
            else if(pivot < med) left = pivot+1;
            else right = pivot-1;
        }
        return med;
    }
    KDTreeNode* buildRecursive(size_t begin, size_t end, int depth = 0) {
        if(end-begin < 1) return nullptr;
        
        const int axis = depth % dim;
        auto comp = [&](size_t a_idx, size_t b_idx) {
            return data[a_idx * dim + axis] < data[b_idx * dim + axis];
        };
        
        // `data` will be changed after `partitionWithMedian`.
        const auto med = partitionWithMedian(begin, end, comp);
        
        auto node = new KDTreeNode();
        
        node->id = dataIdx[med];
        node->axis = axis;
        node->v    = std::vector<T>(data.begin()+med*dim, data.begin()+(med+1)*dim);
        node->l    = buildRecursive(begin, med, depth+1);
        node->r    = buildRecursive(med+1, end, depth+1);
        
        return node;
    }
    void build() {
        if(data.size() < 1) return;
        root = buildRecursive(0, nodeCount());
    }
    
    // Util------------------------------ O(N)
    
    void sequentialDataRecursive(KDTreeNode* node, std::vector<Node<T, dim>*>& collect) {
        if(node == nullptr) return;
        collect.push_back(node);
        sequentialDataRecursive(node->l, collect);
        sequentialDataRecursive(node->r, collect);
    }
    std::vector<Node<T, dim>*> sequentialData() {
        std::vector<Node<T, dim>*> ret;
        sequentialDataRecursive(root, ret);
        return ret;
    }
    
    // Search--------------------------- O(log N)?
    
    T dot(const std::vector<T>& a, const std::vector<T>& b) {
        T ret = 0;
        for(int i = 0; i < a.size(); i++) ret += a[i] * b[i];
        return ret;
    }
    
    void rangeSearchRecursive(KDTreeNode* node, const std::vector<T>& target, const T& range, std::vector<KDTreeNode*>& found) {
        if(node == nullptr) return;
        
        std::vector<T> minDir(target.size()), maxDir(target.size());
        for(int i = 0; i < minDir.size(); i++) {
            minDir[i] = target[i] - range - node->v[i];
            maxDir[i] = target[i] + range - node->v[i];
        }
        T minmax = dot(minDir, maxDir);
        if(minmax < 0) {
            // 만약 내가 포함되는 경우 나를 found에 추가.
            found.push_back(node);
            // 내가 포함되면 내가 가른 양쪽 범위를 모두 확인할 필요 있음.
            rangeSearchRecursive(node->l, target, range, found);
            rangeSearchRecursive(node->r, target, range, found);
        }
        else {
            // 내가 포함되지 않으면, 내가 가른 방향 중 어느 방향이(혹은 둘 다?) 범위에 포함되는지 확인해야 함.
            // 왼쪽 벡터를 node->v에서 range의 최소/최대 지점으로 향하는 벡터와 내적해서 두 부호가 반대일 경우 양쪽 확인 필요. (현재 노드가 범위를 양분함)
            // 두 부호가 같은데 양수면 왼쪽, 음수면 오른쪽 확인.
            std::vector<T> leftDir(node->v.size(), 0);
            leftDir[node->axis] = -1;
            T lmin = dot(leftDir, minDir);
            T lmax = dot(leftDir, maxDir);
            if(lmin > 0 && lmax > 0) {
                rangeSearchRecursive(node->l, target, range, found);
            }
            else if(lmin < 0 && lmax < 0) {
                rangeSearchRecursive(node->r, target, range, found);
            }
            else {
                rangeSearchRecursive(node->l, target, range, found);
                rangeSearchRecursive(node->r, target, range, found);
            }
        }
    }
    
    std::vector<KDTreeNode*> rangeSearch(const std::vector<T>& target, const T& range) {
        std::vector<KDTreeNode*> found;
        rangeSearchRecursive(root, target, range, found);
        return found;
    }
    
    
    std::vector<std::pair<T, KDTreeNode*>> kNearestNeighbors(const std::vector<T>& target, size_t k) {
        if (k == 0 || !root) {
            return {};
        }

        std::priority_queue<std::pair<T, KDTreeNode*>> bestNodes;

        kNNRecursiveSearch(root, target, k, bestNodes);

        std::vector<std::pair<T, KDTreeNode*>> result;
        result.reserve(bestNodes.size());
        while (!bestNodes.empty()) {
            result.emplace_back(bestNodes.top().first, bestNodes.top().second);
            bestNodes.pop();
        }
        std::reverse(result.begin(), result.end());
        
        return result;
    }

    T distanceSq(const std::vector<T>& a, const std::vector<T>& b) const {
        T distSq = 0;
        for (size_t i = 0; i < dim; ++i) {
            T diff = a[i] - b[i];
            distSq += diff * diff;
        }
        return distSq;
    }

    void kNNRecursiveSearch(KDTreeNode* node, const std::vector<T>& target, size_t k,
                            std::priority_queue<std::pair<T, KDTreeNode*>>& bestNodes) {
        if (node == nullptr) return;

        T dSq = distanceSq(node->v, target);

        if (bestNodes.size() < k || dSq < bestNodes.top().first) {
            bestNodes.push({dSq, node});
            if (bestNodes.size() > k) {
                bestNodes.pop();
            }
        }

        int axis = node->axis;
        KDTreeNode* nearChild = (target[axis] < node->v[axis]) ? node->l : node->r;
        KDTreeNode* farChild  = (target[axis] < node->v[axis]) ? node->r : node->l;
        
        kNNRecursiveSearch(nearChild, target, k, bestNodes);

        T distToPlane = target[axis] - node->v[axis];
        distToPlane *= distToPlane; 

        if (bestNodes.size() < k || distToPlane < bestNodes.top().first) {
            kNNRecursiveSearch(farChild, target, k, bestNodes);
        }
    }
    
    
    // Debug---------------------------
    
    void printRecursive(KDTreeNode* node, int depth = 0) {
        if(node == nullptr) return;
        for(int i = 0; i < depth; i++) std::cout << "  ";
        std::cout << "depth "<< depth << ": (" << node->v[0];
        for(int i = 1; i < dim; i++) std::cout << ", " << node->v[i] ;
        std::cout << ")" << std::endl;
        
        printRecursive(node->l, depth+1);
        printRecursive(node->r, depth+1);
    }
    void print() {
        printRecursive(root);
    }
    
    ~KDTree() { delete root; }
};

SparseMatrix<double> A;
VectorXd c;
VectorXd constraints;
VectorXd x;
SparseLU<SparseMatrix<double>> solver;

std::filesystem::path resPath;
std::filesystem::path catPath;
std::filesystem::path lionPath;
std::filesystem::path horsePath;
std::filesystem::path camelPath;

const std::string marker_CatLion = "/marker-cat-lion.txt";
const std::string marker_HorseCamel = "/marker-horse-camel.txt";

std::vector<std::pair<GLuint, GLuint>> markers;
std::vector<int> srcVertexIdxTable;

void readMarkers(const std::string& markerPath) {
    std::ifstream ifs(markerPath);

    if(!ifs.is_open()) {
        std::cerr << "Cannot open " << resPath.string() + marker_CatLion << std::endl;
        return;
    }

    // first = cat, second = lion
    while(!ifs.eof()) {
        GLuint cm, lm;
        ifs >> cm >> lm;
        markers.push_back({cm, lm});
    }
}

vector<vector<unsigned int>> surroundingFaces(const TriangleMesh& mesh, const int nVertices) {
    vector<vector<unsigned int>> perVertices(nVertices);
    for(int i = 0; i < mesh.size(); i++) {
        // if(i % 500 == 0) std::cout << "i: " << i << std::endl;
        perVertices[mesh[i].x()].push_back(i);
        perVertices[mesh[i].y()].push_back(i);
        perVertices[mesh[i].z()].push_back(i);
    }
    return perVertices;
}

vector<vector<unsigned int>> findNeighboringTriangles(const TriangleMesh& mesh, const int nVertices) {
    vector<std::set<unsigned int>> temp(mesh.size());
    vector<vector<unsigned int>> ret(mesh.size(), vector<unsigned int>());

    // helper functions
    auto hasCommonEdge = [](const unsigned int a, const unsigned int b, const Triangle& f) {
        bool aExist = f.x() == a || f.y() == a || f.z() == a;
        bool bExist = f.x() == b || f.y() == b || f.z() == b;
        return aExist && bExist;
    };
    auto isAdjacent = [hasCommonEdge](const Triangle& f1, const Triangle& f2) {
        return hasCommonEdge(f1.x(), f1.y(), f2) || hasCommonEdge(f1.x(), f1.z(), f2) || hasCommonEdge(f1.y(), f1.z(), f2);
    };

    // logic
    vector<vector<unsigned int>> perVertices = surroundingFaces(mesh, nVertices);

    for(int i = 0; i < perVertices.size(); i++) {
        for(int j = 0; j < perVertices[i].size();j++) {
            // perVertices[i]: 점 i를 포함하는 face idx vector
            // perVertices[i][j]: 점 i를 포함하는 j번째 face
            // i를 포함하는 모든 face 중에서, 인접한 face를 찾아서 ret에 넣기.
            // 인접한 face: perVertices[i][j]번 face를 구성하는 세 점 중에 두 점이 겹치는 어떤 perVertices[i][k]가 있다면,
            // 해당 perVertices[i][j]와 perVertices[i][k]는 인접한 face.
            auto jFace = mesh[perVertices[i][j]];
            for(int k = j+1; k < perVertices[i].size(); k++) {
                auto kFace = mesh[perVertices[i][k]];
                if(isAdjacent(jFace, kFace)) {
                    temp[perVertices[i][j]].insert(perVertices[i][k]);
                    temp[perVertices[i][k]].insert(perVertices[i][j]);
                }
            }
        }
    }
    for(int i = 0; i < temp.size(); i++) {
        ret[i].assign(temp[i].begin(), temp[i].end());
    }

    return ret;
}

vector<Triplet<double>> correspondenceTriplets;

vector<ObjData> correspondenceDeformedSrc;
struct CorrespondenceLine {
    GLuint vao, vbo;
    void genBuf(const vector<glm::vec3>& pts) {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(glm::vec3), pts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        std::vector<float> colors;
        for(int i = 0; i < pts.size(); i++) {
            // begin red
            colors.push_back(1);
            colors.push_back(0);
            colors.push_back(0);
            // end green
            colors.push_back(0);
            colors.push_back(1);
            colors.push_back(0);
        }
        GLuint color;
        glGenBuffers(1, &color);
        glBindBuffer(GL_ARRAY_BUFFER, color);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float), colors.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    }
};
vector<CorrespondenceLine> closestValidLine;

double ws = 1.0, wi = 0.001;
void matchCorrespondence(const ObjData& src, const ObjData& tar, const std::string& savelistpath) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    auto start_time = high_resolution_clock::now();

    Vertices srcvertices(src.nVertices);
    for(int i = 0; i < src.nVertices; i++) srcvertices[i] << src.vertices[i].x, src.vertices[i].y, src.vertices[i].z;
    Vertices tarvertices(tar.nVertices);
    for(int i = 0; i < tar.nVertices; i++) tarvertices[i] << tar.vertices[i].x, tar.vertices[i].y, tar.vertices[i].z;

    TriangleMesh srcmesh(src.nElements3);
    for(auto i = 0; i < srcmesh.size(); i++) srcmesh[i] << src.elements3[i].x, src.elements3[i].y, src.elements3[i].z;
    TriangleMesh tarmesh(tar.nElements3);
    for(auto i = 0; i < tarmesh.size(); i++) tarmesh[i] << tar.elements3[i].x, tar.elements3[i].y, tar.elements3[i].z;
    
    std::cout << "Find Neighbors..." << std::endl;
    auto srcAdjTriangles = findNeighboringTriangles(srcmesh, src.nVertices);
    auto srcSurroundingFaces = surroundingFaces(srcmesh, src.nVertices);
    auto tarSurroundingFaces = surroundingFaces(tarmesh, tar.nVertices);

    std::cout << "Build KDTree for Target Mesh" << std::endl;
    vector<double> tarVerData;
    for(int i = 0; i < tar.nVertices; i++) {
        tarVerData.push_back(tar.vertices[i].x);
        tarVerData.push_back(tar.vertices[i].y);
        tarVerData.push_back(tar.vertices[i].z);
    }
    KDTree<double, 3> tarKDTree(tarVerData);

    // T=\tilde V V^{-1} : 3 by 3 matrix
    // wanna make T to Ax form where A : 9 by 9 and x : 9 by 1.
    // x collects \tilde V's columns in a column.
    // A_{ij}=diag(V_{ij}) : 3 by 3 matrix
    // TtoAx add the triplets for the sparse matrix A.
    auto TtoAx = [](const Vertices& vertices, const Triangle& face, const int fid, const int row, const double weight, vector<Triplet<double>>& triplets) {
        int idx1 = face.x();
        int idx2 = face.y();
        int idx3 = face.z();
        int idx4 = static_cast<int>(vertices.size()+fid);
        auto p1 = vertices[idx1];
        auto p2 = vertices[idx2];
        auto p3 = vertices[idx3];
        auto p4 = p1 + (p2-p1).cross(p3-p1).normalized();

        Matrix3d V;
        V.col(0) = p2-p1;
        V.col(1) = p3-p1;
        V.col(2) = p4-p1;
        V = V.inverse()*weight;

        for(int col = 0; col < 3; col++) for(int diag = 0; diag < 3; diag++) {
            triplets.push_back({row+col*3+diag, idx1*3+diag, -V.col(col).sum()});
            triplets.push_back({row+col*3+diag, idx2*3+diag, V.coeff(0, col)});
            triplets.push_back({row+col*3+diag, idx3*3+diag, V.coeff(1, col)});
            triplets.push_back({row+col*3+diag, idx4*3+diag, V.coeff(2, col)});
        }
    };

    // find the closest valid point on target mesh for each x
    auto isValid = [](const Vector3d& vn, const Triangle& f, const Vertices& vertices) {
        auto a = vertices[f.x()];
        auto b = vertices[f.y()];
        auto c = vertices[f.z()];
        auto fn = (b-a).cross(c-a);
        return vn.dot(fn) >= 0;
    };

    // https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d
    auto closestPointOnTriangle = [](const Vector3d& p, const Vector3d& a, const Vector3d& b, const Vector3d& c) -> Vector3d {
        Vector3d ab = b - a;
        Vector3d ac = c - a;
        Vector3d ap = p - a;

        double d1 = ab.dot(ap);
        double d2 = ac.dot(ap);
        if (d1 <= 0.0 && d2 <= 0.0) return a; // vertex a

        Vector3d bp = p - b;
        double d3 = ab.dot(bp);
        double d4 = ac.dot(bp);
        if (d3 >= 0.0 && d4 <= d3) return b; // vertex b

        Vector3d cp = p - c;
        double d5 = ab.dot(cp);
        double d6 = ac.dot(cp);
        if (d6 >= 0.0 && d5 <= d6) return c; // vertex c

        double vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
            double v = d1 / (d1 - d3);
            return a + v * ab; // edge ab
        }

        double vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
            double w = d2 / (d2 - d6);
            return a + w * ac; // edge ac
        }

        double va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
            double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + w * (c - b); // edge bc
        }

        double denom = 1.0 / (va + vb + vc);
        double v = vb * denom;
        double w = vc * denom;
        return a + v * ab + w * ac; // face
    };

    std::cout << "x set" << std::endl;
    x.setZero((srcvertices.size()+srcmesh.size())*3);
    std::cout << "constraints set" << std::endl;
    constraints.setZero(x.size());
    for(const auto& m : markers) {
        constraints(m.first*3  ) = tar.vertices[m.second].x;
        constraints(m.first*3+1) = tar.vertices[m.second].y;
        constraints(m.first*3+2) = tar.vertices[m.second].z;
    }
    vector<double> wc = {0, 1, 10, 100, 1000, 5000};
    for(int wciter = 0; wciter < wc.size(); wciter++) {
        closestValidLine.resize(wc.size()-1);
        std::cout << "Iteration " << wciter << " Start..." << std::endl;
        correspondenceTriplets.clear();

        // solve optimization
        // term 1. smoothness
        std::cout << "  Smoothness Term Compute..." << std::endl;

        int rowCount = 0;
        double sws = sqrt(ws);
        for(int i = 0; i < srcmesh.size(); i++) {
            for(int adj = 0; adj < srcAdjTriangles[i].size(); adj++) {
                auto j = srcAdjTriangles[i][adj];
                TtoAx(srcvertices, srcmesh[i], i, rowCount, sws,  correspondenceTriplets);
                TtoAx(srcvertices, srcmesh[j], j, rowCount, -sws, correspondenceTriplets);
                rowCount+=9;
            }
        }

        // term 2. identity
        // E_I = \sigma_i^{|T|}{ \| I - T_i \|_F^2 }
        std::cout << "  Identity Term Compute..." << std::endl;

        vector<std::pair<int, double>> cTerm_Identity;
        // double swi = sqrt(wi);
        double swi = wi;
        for(int i = 0; i < srcmesh.size(); i++) {
            TtoAx(srcvertices, srcmesh[i], i, rowCount, swi, correspondenceTriplets);

            cTerm_Identity.push_back({rowCount,   swi});
            cTerm_Identity.push_back({rowCount+4, swi});
            cTerm_Identity.push_back({rowCount+8, swi});

            rowCount+=9;
        }

        auto updateNormal = [srcvertices](const Vertices& vertices, const TriangleMesh& mesh) {
            VectorXd normals(vertices.size()*3);
            normals.setZero();
            for(const auto& face : mesh) {
                auto a = srcvertices[face.x()];
                auto b = srcvertices[face.y()];
                auto c = srcvertices[face.z()];

                auto n = (b-a).cross(c-a);
                normals.segment<3>(3*face.x()) += n;
                normals.segment<3>(3*face.y()) += n;
                normals.segment<3>(3*face.z()) += n;
            }
            for(int i = 0; i < normals.size(); i += 3) normals.segment<3>(i).normalize();
            return normals;
        };

        // term 3. closest valid point
        vector<std::pair<int, double>> cTerm_ClosestValidPoint;
        if(wciter != 0) {
            std::cout << "  ClosestValidPoint Term Compute... weight: " << wc[wciter] << " and sqrt weightL: " << sqrt(wc[wciter]) << std::endl;

            // compute vertex normals of x;
            VectorXd normals = updateNormal(srcvertices, srcmesh);

            // compute closest valid point
            // double swc = sqrt(wc[wciter]);
            double swc = wc[wciter];
            vector<glm::vec3> closestValidPts;
            for(int i = 0; i < normals.size(); i += 3) {
                auto p = srcvertices[i/3];
                vector<double> validPoint;
                int k = 20;
                while(validPoint.empty() && k <= tar.nVertices) {
                    auto nn = tarKDTree.kNearestNeighbors({p.x(), p.y(), p.z()}, k);
                    std::set<unsigned int> validTriangleSet;
                    for(const auto& nei : nn) {
                        auto surr = tarSurroundingFaces[nei.second->id];
                        for(auto s : surr) {
                            if(isValid(normals.segment<3>(i), tarmesh[s], tarvertices)) {
                                validTriangleSet.insert(s);
                            }
                        }
                    }

                    double curMinDist = -1.0;
                    for(const auto& fid : validTriangleSet) {
                        auto cp = closestPointOnTriangle(p, tarvertices[tarmesh[fid].x()], tarvertices[tarmesh[fid].y()], tarvertices[tarmesh[fid].z()]);
                        double curDist = (cp-p).squaredNorm();
                        if(curMinDist < 0 || curDist < curMinDist) {
                            curMinDist = curDist;
                            if(validPoint.empty()) validPoint.resize(3);
                            validPoint[0] = cp.x();
                            validPoint[1] = cp.y();
                            validPoint[2] = cp.z();
                        }
                    }
                    k *= 2;
                    // if(validPoint.empty()) std::cout << i << "-th vertex k expand: k = " << k << std::endl;
                }
                if(validPoint.empty()) std::cout << i << "-th vertex finding validpoint failed." << std::endl;
                correspondenceTriplets.push_back({rowCount,   i,   swc});
                correspondenceTriplets.push_back({rowCount+1, i+1, swc});
                correspondenceTriplets.push_back({rowCount+2, i+2, swc});

                cTerm_ClosestValidPoint.push_back({rowCount,   swc*validPoint[0]});
                cTerm_ClosestValidPoint.push_back({rowCount+1, swc*validPoint[1]});
                cTerm_ClosestValidPoint.push_back({rowCount+2, swc*validPoint[2]});
                rowCount+=3;
                closestValidPts.push_back({srcvertices[i/3].x(), srcvertices[i/3].y(), srcvertices[i/3].z()});
                closestValidPts.push_back({validPoint[0], validPoint[1], validPoint[2]});

                if(i % 3000 == 0) std::cout << "    " << i/3 << " vertices are processed." << std::endl;
            }
            closestValidLine[wciter-1].genBuf(closestValidPts);
        }

        // Sparse LU Solver
        std::cout << "  Correspondence System Construct..." << std::endl;

        std::cout << "    c set zero" << std::endl;
        c.setZero(rowCount);
        std::cout << "    c put values - identity(" << cTerm_Identity.size() << ")" << std::endl;
        for(int i = 0; i < cTerm_Identity.size(); i++) c(cTerm_Identity[i].first) = cTerm_Identity[i].second;
        std::cout << "    c put values - closestvalidpoint(" << cTerm_ClosestValidPoint.size() << ")" << std::endl;
        for(int i = 0; i < cTerm_ClosestValidPoint.size(); i++) c(cTerm_ClosestValidPoint[i].first) = cTerm_ClosestValidPoint[i].second;

        std::cout << "    A put values" << std::endl;
        A = SparseMatrix<double>(c.size(), x.size());
        A.setFromTriplets(correspondenceTriplets.begin(), correspondenceTriplets.end());
        std::cout << "  System: A(" << A.rows() << ", " << A.cols() << "), x(" << x.rows() << ", " << x.cols() << "), c(" << c.rows() << ", " << c.cols() << ")" << std::endl;
        c = c - A*constraints;

        SparseMatrix<double> B = A.transpose() * A;
        VectorXd d = A.transpose() * c;

        // Induce the constraints on the system
        for (const auto& m : markers) {
            for (int i = 0; i < 3; ++i) {
                int var_idx = m.first * 3 + i;

                B.prune([var_idx](int row, int col, double value) { return (row != var_idx) && (col != var_idx); });

                B.coeffRef(var_idx, var_idx) = 1.0;

                d(var_idx) = constraints(var_idx);
            }
        }

        std::cout << "  System Analyze and Start Solve..." << std::endl;
        // 수정된 시스템으로 방정식을 풉니다.
        auto aptimes = high_resolution_clock::now();
        solver.analyzePattern(B);
        auto aptimee = std::chrono::high_resolution_clock::now();
        std::cout << "  Analyzed in " << duration_cast<milliseconds>(aptimee-aptimes).count()/1000.0 << "s" << std::endl;
        auto ftimes = high_resolution_clock::now();
        solver.factorize(B);
        auto ftimee = high_resolution_clock::now();
        std::cout << "  Factorized in " << duration_cast<milliseconds>(ftimee-ftimes).count()/1000.0 << "s" << std::endl;
        auto stimes = high_resolution_clock::now();
        x = solver.solve(d);
        auto stimee = high_resolution_clock::now();
        std::cout << "  Solved in " << duration_cast<milliseconds>(stimee-stimes).count()/1000.0 << "s" << std::endl;
        std:: cout<< "Iteration " << wciter << " solved." << std::endl;

        for(int i = 0; i < src.nVertices; i++) srcvertices[i] = x.segment<3>(i*3);
        correspondenceDeformedSrc.push_back(src);
        for(int i = 0; i < src.nVertices; i++) correspondenceDeformedSrc.rbegin()->vertices[i] = {srcvertices[i].x(), srcvertices[i].y(), srcvertices[i].z()};
        auto normals = updateNormal(srcvertices, srcmesh);
        for(int i = 0; i < src.nVertices; i++) correspondenceDeformedSrc.rbegin()->syncedNormals[i] = {normals(i*3), normals(i*3+1), normals(i*3+2)};
    }

    auto end_time = high_resolution_clock::now();
    auto duration = end_time - start_time;
    std::cout << "--- Correspondence System Solved in ";
    std::cout << duration_cast<milliseconds>(duration).count()/1000.0 << "s! ---" << std::endl;;


    // generate correspondence list
    std::cout << "--- Find Correspondence Pair ---" << std::endl;
    std::set<std::pair<unsigned int, unsigned int>> correspondencePairSet;
    // double sf = tar.scale.x > tar.scale.y ? tar.scale.x > tar.scale.z ? tar.scale.x : tar.scale.z : tar.scale.y > tar.scale.z ? tar.scale.y : tar.scale.z;
    // const double threshold = sf*0.001;
    auto computeThreshold = [](const VectorXd& obj, const int numFace) {
        const double BIG = 987654321;
        Vector3d max(-BIG, -BIG, -BIG), min(BIG, BIG, BIG);
        for(int i = 0; i < obj.size(); i+=3) {
            Vector3d p = obj.segment<3>(i);
            min.x() = p.x() < min.x() ? p.x() : min.x();
            min.y() = p.y() < min.y() ? p.y() : min.y();
            min.z() = p.z() < min.z() ? p.z() : min.z();
            max.x() = p.x() > max.x() ? p.x() : max.x();
            max.y() = p.y() > max.y() ? p.y() : max.y();
            max.z() = p.z() > max.z() ? p.z() : max.z();
        }
        Vector3d scale = max-min;
        return sqrt(4 * (scale.x()*scale.y() + scale.x()*scale.z() + scale.y()*scale.z()) / numFace);
    };
    VectorXd tarx(srcvertices.size()*3);
    for(int i = 0; i < srcvertices.size(); i++) tarx.segment<3>(i*3) = srcvertices[i];
    const double threshold = std::max(computeThreshold(x, srcmesh.size()), computeThreshold(tarx, tarmesh.size()));
    std::cout << "  Threshold: " << threshold << std::endl;
    std::cout << "  src to tar.. " << std::endl;
    for(int i = 0; i < srcmesh.size(); i++) {
        // calc source face normal
        auto face = srcmesh[i];
        auto sa = srcvertices[face.x()];
        auto sb = srcvertices[face.y()];
        auto sc = srcvertices[face.z()];
        auto fn = (sb-sa).cross(sc-sa).normalized();
        auto sCentroid = (sa+sb+sc)/3.0;
        
        // calc close target faces(triangles)
        std::set<unsigned int> candTriSet;
        auto findTriFromKNN = [&tarKDTree, &candTriSet, &tarSurroundingFaces, &threshold](const Vector3d& p) {
            // int k = 20;
            // auto nn = tarKDTree.kNearestNeighbors({p.x(), p.y(), p.z()}, k);
            auto nn = tarKDTree.rangeSearch({p.x(), p.y(), p.z()}, threshold);
            for (const auto& n : nn) {
                auto surr = tarSurroundingFaces[n->id];
                for(const auto& s : surr) {
                    candTriSet.insert(s);
                }
            }
        };
        findTriFromKNN(sa);
        findTriFromKNN(sb);
        findTriFromKNN(sc);

        // validity check
        double minDist = -1.0;
        std::pair<unsigned int, unsigned int> closestPair;
        for(const auto& candTri : candTriSet) {
            auto candFace = tarmesh[candTri];
            auto candA = tarvertices[candFace.x()];
            auto candB = tarvertices[candFace.y()];
            auto candC = tarvertices[candFace.z()];
            auto candFn = (candB-candA).cross(candC-candA).normalized();
            auto candCentroid = (candA+candB+candC)/3.0;
            auto diff = sCentroid-candCentroid;
            auto distSq = diff.dot(diff);
            // condition 1. new dist is in range of threshold
            // condition 2. two faces are facing same direction
            // condition 3. new face or new better face
            if(distSq < threshold && fn.dot(candFn) > 0 && (distSq < minDist || minDist < 0)) {
                minDist = distSq;
                closestPair.first = i;
                closestPair.second = candTri;
            }
        }
        if(minDist > 0) correspondencePairSet.insert(closestPair);

        if(i % 1000 == 0) std::cout << "    " << i << "-th face processed..." << std::endl;
    }
    vector<double> srcVerData;
    for(int i = 0; i < src.nVertices; i++) {
        srcVerData.push_back(srcvertices[i].x());
        srcVerData.push_back(srcvertices[i].y());
        srcVerData.push_back(srcvertices[i].z());
    }
    KDTree<double, 3> srcKDTree(srcVerData);
    std::cout << "  tar to src.. " << std::endl;
    for(int i = 0; i < tarmesh.size(); i++) {
        // calc target face normal
        auto face = tarmesh[i];
        auto ta = tarvertices[face.x()];
        auto tb = tarvertices[face.y()];
        auto tc = tarvertices[face.z()];
        auto fn = (tb-ta).cross(tc-ta).normalized();
        auto tCentroid = (ta+tb+tc)/3.0;
        
        // calc close target faces(triangles)
        std::set<unsigned int> candTriSet;
        auto findTriFromKNN = [&srcKDTree, &candTriSet, &srcSurroundingFaces, threshold](const Vector3d& p) {
            // int k = 20;
            // auto nn = srcKDTree.kNearestNeighbors({p.x(), p.y(), p.z()}, k);
            auto nn = srcKDTree.rangeSearch({p.x(), p.y(), p.z()}, threshold);
            for (const auto& n : nn) {
                auto surr = srcSurroundingFaces[n->id];
                for(const auto& s : surr) {
                    candTriSet.insert(s);
                }
            }
        };
        findTriFromKNN(ta);
        findTriFromKNN(tb);
        findTriFromKNN(tc);

        // validity check
        double minDist = -1.0;
        std::pair<unsigned int, unsigned int> closestPair;
        for(const auto& candTri : candTriSet) {
            auto candFace = srcmesh[candTri];
            auto candA = srcvertices[candFace.x()];
            auto candB = srcvertices[candFace.y()];
            auto candC = srcvertices[candFace.z()];
            auto candFn = (candB-candA).cross(candC-candA).normalized();
            auto candCentroid = (candA+candB+candC)/3.0;
            auto diff = tCentroid-candCentroid;
            auto distSq = diff.dot(diff);
            if(distSq < threshold && fn.dot(candFn) > 0 && (distSq < minDist || minDist < 0)) {
                minDist = distSq;
                closestPair.second = i;
                closestPair.first = candTri;
            }
        }
        if(minDist > 0) correspondencePairSet.insert(closestPair);
        if(i % 1000 == 0) std::cout << "    " << i << "-th face processed..." << std::endl;
    }
    std::cout << "  Found " << correspondencePairSet.size() << " Pairs!" << std::endl;
    
    // save correspondence list
    std::ofstream ofs(resPath / savelistpath.c_str(), std::ofstream::trunc);
    for(const auto& cpair : correspondencePairSet) ofs << cpair.first << " " << cpair.second << std::endl;
    ofs.close();
    std::cout << "File saved in " << (resPath / savelistpath) << std::endl;
}

vector<CorrespondenceLine> cLines;

int viewmode = 0;
float defOffset = 0;
bool deformed = true;
void keycallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if(key == GLFW_KEY_0 && action == GLFW_PRESS) {
        viewmode = 0;
    }
    if(key == GLFW_KEY_1 && action == GLFW_PRESS) {
        viewmode = 1;
    }
    if(key == GLFW_KEY_2 && action == GLFW_PRESS) {
        viewmode = 2;
    }
    if(key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
        defOffset -= 0.3;
        std::cout << "Move left: " << defOffset << std::endl;
    }
    if(key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
        defOffset += 0.3;
        std::cout << "Move right: " << defOffset << std::endl;
    }
    if(key == GLFW_KEY_D && action == GLFW_PRESS) {
        deformed = !deformed;
    }
}

void prepareCorrespondence(const std::string& srcObjParentPath, const std::string& srcObjFile, 
                           const std::string& tarObjParentPath, const std::string& tarObjFile, 
                           const std::string& markerPath, const std::string& correspondencePath) {
    srcOriginal.loadObject(srcObjParentPath.c_str(), srcObjFile);
    tarOriginal.loadObject(tarObjParentPath.c_str(), tarObjFile);

    readMarkers(resPath.string() + markerPath);

    matchCorrespondence(srcOriginal, tarOriginal, correspondencePath);

    srcOriginal.generateBuffers();
    tarOriginal.generateBuffers();

    for(auto& ds : correspondenceDeformedSrc) ds.generateBuffers();

    cLines.resize(correspondenceDeformedSrc.size());
    for(int c = 0; c < cLines.size(); c++) {
        vector<glm::vec3> lines;
        for(int i = 0; i < srcOriginal.nVertices; i++) {
            lines.push_back(srcOriginal.vertices[i]);
            lines.push_back(correspondenceDeformedSrc[c].vertices[i]);
        }

        cLines[c].genBuf(lines);
    }
}
ObjData srcDeformed;
ObjData tarDeformed;

void deformationTransfer(const std::string& srcObjParentPath, const std::string& srcObjFile, 
                         const std::string& srcDeformedObjParentPath, const std::string& srcDeformedObjFile, 
                         const std::string& tarObjParentPath, const std::string& tarObjFile, 
                         const std::string& correspondencePath) {
    // load objects
    srcOriginal.loadObject(srcObjParentPath.c_str(), srcObjFile);
    srcDeformed.loadObject(srcDeformedObjParentPath.c_str(), srcDeformedObjFile);
    tarOriginal.loadObject(tarObjParentPath.c_str(), tarObjFile);
    std::cout << "--- Objects loaded ---" << std::endl;

    // read correspondence set.
    std::ifstream ifs(resPath / correspondencePath);
    std::set<std::pair<unsigned int, unsigned int>> cPairs;
    while(!ifs.eof()) {
        unsigned int srcFaceIdx, tarFaceIdx;
        ifs >> srcFaceIdx >> tarFaceIdx;
        cPairs.insert({srcFaceIdx, tarFaceIdx});
    }
    std::cout << "--- Correspondence set file has been read. (" << cPairs.size() <<" pairs) ---" << std::endl;

    // build system
    std::cout << "--- Build system ---" << std::endl;
    A = SparseMatrix<double>(9*cPairs.size(), (tarOriginal.nVertices+tarOriginal.nElements3)*3);
    c = VectorXd(9*cPairs.size());

    int row = 0;
    vector<Triplet<double>> deformationTransferTriplets;
    auto getMat = [](const ObjData& obj, const unsigned int fid) {
        int idx1 = obj.elements3[fid].x;
        int idx2 = obj.elements3[fid].y;
        int idx3 = obj.elements3[fid].z;
        int idx4 = obj.nVertices+fid;
        
        Vector3d v1 = Vector3d(obj.vertices[idx1].x, obj.vertices[idx1].y, obj.vertices[idx1].z);
        Vector3d v2 = Vector3d(obj.vertices[idx2].x, obj.vertices[idx2].y, obj.vertices[idx2].z);
        Vector3d v3 = Vector3d(obj.vertices[idx3].x, obj.vertices[idx3].y, obj.vertices[idx3].z);
        Vector3d v4 = (v2-v1).cross(v3-v1).normalized();
        Matrix3d V;
        V.col(0) = v2-v1;
        V.col(1) = v3-v1;
        V.col(2) = v4-v1;

        return V;
    };
    std::cout << "  Start make triplets..." << std::endl;
    for (const auto& cp : cPairs) {
        // c
        Matrix3d S = getMat(srcOriginal, cp.first);
        Matrix3d Shat = getMat(srcDeformed, cp.first);

        S = Shat * S.inverse();

        // c.segment<3>(row)   = S.col(0);
        // c.segment<3>(row+3) = S.col(1);
        // c.segment<3>(row+6) = S.col(2);
        for(int r = 0; r < 9; r++) c(row+r) = S.coeff(r%3, r/3);

        // A
        Matrix3d T = getMat(tarOriginal, cp.second);
        T = T.inverse().eval();

        int idx1 = tarOriginal.elements3[cp.second].x;
        int idx2 = tarOriginal.elements3[cp.second].y;
        int idx3 = tarOriginal.elements3[cp.second].z;
        int idx4 = tarOriginal.nVertices+cp.second;

        for(int col = 0; col < 3; col++) for(int diag = 0; diag < 3; diag++) {
            deformationTransferTriplets.push_back({row+col*3+diag, idx1*3+diag, -T.col(col).sum()});
            deformationTransferTriplets.push_back({row+col*3+diag, idx2*3+diag,  T.coeff(0, col)});
            deformationTransferTriplets.push_back({row+col*3+diag, idx3*3+diag,  T.coeff(1, col)});
            deformationTransferTriplets.push_back({row+col*3+diag, idx4*3+diag,  T.coeff(2, col)});
        }
        row+= 9;
    }
    std::cout << "  End make triplets" << std::endl;

    A.setFromTriplets(deformationTransferTriplets.begin(), deformationTransferTriplets.end());
    std::cout << "--- Linear system constructed: A(" << A.rows() << ", " << A.cols() << "), c(" << c.size() << ") ---" << std::endl;

    int notmatched = 0;
    for(int i = 0; i < A.cols(); i+=3) {
        if(A.col(i).nonZeros() == 0) notmatched++;
    }
    std::cout << "  Not matched points: " << notmatched << std::endl;

    auto ATA = A.transpose() * A;
    auto ATc = A.transpose() * c;
    std::cout << "--- Linear system constructed: A^TA(" << ATA.rows() << ", " << ATA.cols() << "), A^Tc(" << ATc.size() << ") ---" << std::endl;

    std::cout << "--- Start to solve system ---" << std::endl;
    solver.compute(ATA);
    std::cerr << "  State error on compute: " << solver.info() << std::endl;
    std::cout << "  System factorized" << std::endl;
    x = solver.solve(ATc);
    std::cout << "--- System solved ---" << std::endl;

    tarDeformed = tarOriginal;
    const double big = 987654321;
    double minY = big;
    double maxX = -big, minX = big; 
    double maxZ = -big, minZ = big;
    for(int i = 0; i < tarDeformed.nVertices; i++) {
        minX = minX > x(i*3) ? x(i*3) : minX;
        maxX = maxX < x(i*3) ? x(i*3) : maxX;
        minY = minY > x(i*3+1) ? x(i*3+1) : minY;
        minZ = minZ > x(i*3+2) ? x(i*3+2) : minZ;
        maxZ = maxZ < x(i*3+2) ? x(i*3+2) : maxZ;
    }
    double centerX = (maxX+minX)/2;
    double centerZ = (maxZ+minZ)/2;
    for(int i = 0; i < tarDeformed.nVertices; i++) tarDeformed.vertices[i] = {x(i*3)-centerX, x(i*3+1)-minY, x(i*3+2)-centerZ};
    
    vector<Vector3d> normals(tarDeformed.nSyncedNormals);
    for(int i = 0; i < tarDeformed.nElements3; i++) {
        auto face = tarDeformed.elements3[i];
        auto a = tarDeformed.vertices[face.x];
        auto b = tarDeformed.vertices[face.y];
        auto c = tarDeformed.vertices[face.z];
        auto n = glm::cross(b-a, c-a);
        auto nn = Vector3d(n.x, n.y, n.z);
        normals[face.x] += nn;
        normals[face.y] += nn;
        normals[face.z] += nn;
    }
    for(int i = 0; i < normals.size(); i++) {
        auto nn = normals[i].normalized();
        tarDeformed.normals[i] = {nn.x(), nn.y(), nn.z()};
    }
    std::cout << "--- Transfer end ---" << std::endl;
    srcOriginal.generateBuffers();
    srcDeformed.generateBuffers();
    tarOriginal.generateBuffers();
    tarDeformed.generateBuffers();
}

void init() {
    // set default settings
    shader.loadShader((resPath.string() + "/test.vert").c_str(), (resPath.string() + "/test.frag").c_str());
    camera.setPosition({0, 0, 10});
    camera.look = glm::vec3(0, -0.5f, 0);
    camera.glfwSetCallbacks(window->getGLFWWindow());
    glEnable(GL_DEPTH_TEST);
    glfwSetKeyCallback(window->getGLFWWindow(), keycallback);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);



    // prepareCorrespondence(catPath.string(), "cat-reference.obj", lionPath.string(), "lion-reference.obj", marker_CatLion, "correspondence-cat-lion.txt");
    // prepareCorrespondence(horsePath.string(), "horse-gallop-reference.obj", camelPath.string(), "camel-gallop-reference.obj", marker_HorseCamel, "correspondence-horse-camel.txt");
//    deformationTransfer(catPath.string(), "cat-reference.obj", catPath.string(), "cat-01.obj", lionPath.string(), "lion-reference.obj",  "correspondence-cat-lion.txt");
     deformationTransfer(horsePath.string(), "horse-gallop-reference.obj", horsePath.string(), "horse-gallop-01.obj", camelPath.string(), "camel-gallop-reference.obj",  "correspondence-horse-camel.txt");


    glErr("after init");
}

void render() {
    glViewport(0, 0, window->width(), window->height());
    glClearColor(0, 0, 0.3f, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    
    // cat render
    glm::mat4 M(1);
    M = glm::rotate(glm::translate(glm::mat4(1), glm::vec3(-1.5, 0, 0)), PI/2.f, glm::vec3(0, 1, 0));
    glm::mat4 V = camera.lookAt();
    glm::mat4 P = camera.perspective(window->aspect(), 0.1f, 1000.f);
    glm::mat4 MV = V * M;
    shader.use();
    shader.setUniform("mode", 0);
    shader.setUniform("NormalMat", glm::mat3(MV[0], MV[1], MV[2]));
    shader.setUniform("V", V);
    shader.setUniform("MV", MV);
    shader.setUniform("MVP", P * MV);
    glErr("after set uniforms");
    srcOriginal.render();
    glErr("after render a srcOriginal");
    

    // lion render
    M = glm::mat4(1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,    0.3, 0, 0, 1);
    M = glm::rotate(glm::translate(glm::mat4(1), glm::vec3(-0.5, 0, 0)), PI/2.f, glm::vec3(0, 1, 0));
    V = camera.lookAt();
    P = camera.perspective(window->aspect(), 0.1f, 1000.f);
    MV = V * M;
    shader.setUniform("NormalMat", glm::mat3(MV[0], MV[1], MV[2]));
    shader.setUniform("V", V);
    shader.setUniform("MV", MV);
    shader.setUniform("MVP", P * MV);
    tarOriginal.render();
    glErr("after render a tarOriginal");


    // src deformed
    M = glm::mat4(1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,    0.6, 0, 0, 1);
    M = glm::rotate(glm::translate(glm::mat4(1), glm::vec3(0.5, 0, 0)), PI/2.f, glm::vec3(0, 1, 0));
    V = camera.lookAt();
    P = camera.perspective(window->aspect(), 0.1f, 1000.f);
    MV = V * M;
    shader.setUniform("NormalMat", glm::mat3(MV[0], MV[1], MV[2]));
    shader.setUniform("V", V);
    shader.setUniform("MV", MV);
    shader.setUniform("MVP", P * MV);
    srcDeformed.render();
    glErr("after render a srcDeformed");


    // tar deformed
    M = glm::mat4(1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,    0.9, 0, 0, 1);
    M = glm::rotate(glm::translate(glm::mat4(1), glm::vec3(1.5, 0, 0)), PI/2.f, glm::vec3(0, 1, 0));
    V = camera.lookAt();
    P = camera.perspective(window->aspect(), 0.1f, 1000.f);
    MV = V * M;
    shader.setUniform("NormalMat", glm::mat3(MV[0], MV[1], MV[2]));
    shader.setUniform("V", V);
    shader.setUniform("MV", MV);
    shader.setUniform("MVP", P * MV);
    tarDeformed.render();
    glErr("after render a tarDeformed");

    // deformedCat redner
    float offset = defOffset;
    for(int i = 0; i < correspondenceDeformedSrc.size(); i++) {
        M = glm::mat4(1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,    offset, -0.5, 0, 1);
        V = camera.lookAt();
        P = camera.perspective(window->aspect(), 0.1f, 1000.f);
        MV = V * M;
        shader.setUniform("NormalMat", glm::mat3(MV[0], MV[1], MV[2]));
        shader.setUniform("V", V);
        shader.setUniform("MV", MV);
        shader.setUniform("MVP", P * MV);
        offset += 0.3;
        if(viewmode == 0) {
            if(deformed) correspondenceDeformedSrc[i].render();
            else tarOriginal.render();
        }
        else if(viewmode == 1) {
            shader.setUniform("mode", 2);
            glBindVertexArray(cLines[i].vao);
            glDrawArrays(GL_LINES, 0, correspondenceDeformedSrc[i].nVertices*2);
        }
        else if(viewmode == 2) {
            if(i == 0) continue;
            shader.setUniform("mode", 2);
            glBindVertexArray(closestValidLine[i-1].vao);
            glDrawArrays(GL_LINES, 0, correspondenceDeformedSrc[i].nVertices*2);
        }
        glErr("after render a deformedCat");

    }

    
}

int main(int argc, char *argv[]) {
    std::filesystem::path exePath = argv[0];
    std::filesystem::path buildPath = exePath.parent_path();
    resPath = buildPath / "resources";
    catPath = resPath / "cat-poses";
    lionPath = resPath / "lion-poses";
    camelPath = resPath / "camel-gallop";
    horsePath = resPath / "horse-gallop";

    window = new YGLWindow(640, 480, "Deformation Transfer For Triangle Meshes");
    window->mainLoop(init, render);

    return 0;
}


