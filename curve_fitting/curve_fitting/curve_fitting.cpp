#include <iostream>
#include <Eigen/Core>

#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>
using namespace std;

// 曲線のモデル y=ax^2+bx+c
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	virtual void setToOriginImpl()
	{
		_estimate << 0, 0, 0;
	}

	virtual void oplusImpl(const double *update)
	{
		_estimate += Eigen::Vector3d(update);
	}

	// slambookの通りに中身書かずに実行するとエラーが出た
	// g2oのsample見ると下のようになっていたのでそちらを真似た
	virtual bool read(istream &in)
	{
		cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
		return false;
	}
	virtual bool write(ostream &out) const 
	{
		cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
		return false;
	}
};

// 誤差
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	double _x;
	CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}
	void computeError()
	{
		const CurveFittingVertex *v = static_cast<const CurveFittingVertex*> (_vertices[0]);
		const Eigen::Vector3d abc = v->estimate();
		_error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
	}
	virtual bool read(istream &in) 
	{
		cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
		return false;
	}
	virtual bool write(ostream &out) const
	{
		cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
		return false;
	}
};

int main()
{
	double a = 5.0, b = -4.0, c = 7.0;  // モデル真値
	int N = 100;  // データ数
	double abc[3] = { 0, 0, 0 };

	// generate data
	double w_sigma = 1.0;
	cv::RNG rng;
	vector<double> x_data, y_data;
	for (int i = 0; i < N; i++)
	{
		double x = i / double(N);
		x_data.push_back(x);
		y_data.push_back(exp(a * x * x + b * x + c) + rng.gaussian(w_sigma));
	}

	typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
	typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
	typedef g2o::OptimizationAlgorithmLevenberg LM;
	typedef g2o::OptimizationAlgorithmGaussNewton GN;

	auto solver = new LM(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(true);

	CurveFittingVertex *v = new CurveFittingVertex();
	v->setEstimate(Eigen::Vector3d(abc));
	v->setId(0);
	optimizer.addVertex(v);

	for (int i = 0; i < N; i++)
	{
		CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
		edge->setId(i);
		edge->setVertex(0, v);
		edge->setMeasurement(y_data[i]);
		edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
		optimizer.addEdge(edge);
	}

	cout << "start optimization" << endl;
	auto t1 = chrono::steady_clock::now();
	optimizer.initializeOptimization();
	optimizer.optimize(10);  // iteration回数
	auto t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "solve time cost = " << time_used.count() << " second. " << endl;

	Eigen::Vector3d abc_estimate = v->estimate();
	cout << "ground truth : " << a << " " << b << " " << c << endl;
	cout << "estimation : " << abc_estimate.transpose() << endl;

	return 0;
}