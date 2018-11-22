/// copied from one_layer_NN (2018 11 02)
/// needs modification

#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>
typedef unsigned int uint;
typedef double(*ActFunc)(double);

class Network
{
private:
	class Weight;

	double *input;
	uint inlen;

	uint hiddenlen;
	uint *hiddensize;

	double **z;
	double **l;
	double *output;
	uint outlen;

	double alpha;

	ActFunc **activationFunction;
	ActFunc **dactivationFunction;

	static double zero(double x) { return 0; }
	static double one(double x) { return 1; }
	static double identity(double x) { return x; }
	static double sigmoid(double x) { return 1 / (1 + exp(-x)); }
	static double arctan(double x) { return atan(x); }
	static double tanhyp(double x) { return tanh(x); }
	static double relu(double x) { return x > 0 ? x : 0; }

	static double dsigmoid(double x) { double t = sigmoid(x); return t * (1 - t); }
	static double darctan(double x) { double t = atan(x); return 1 / (t * t + 1); }
	static double dtanhyp(double x) { double t = tanh(x); return 1 - t * t; }
	static double drelu(double x) { return x > 0 ? 1 : (x < 0 ? 0 : 0.5); }

	void forward(const Weight &weight, const double *const input, const uint &inputlen, double **const output, const uint &outputlen);
	void update(const Weight &weight, const double *const delta);

	class Weight
	{
	private:
		double **w;
		uint row;
		uint col;

	public:
		Weight(const uint &inlen = 0, const uint &outlen = 0, const double &b = 0)
			:row(inlen + 1), col(outlen), w(new double*[inlen + 1])
		{
			std::random_device rd;
			std::mt19937_64 rnd(rd());
			std::normal_distribution<double> distribution(0, 1);
			w[0] = new double[outlen]();
			for (uint i = 1; i <= inlen; i++)
			{
				w[i] = new double[outlen];
				for (uint j = 0; j < outlen; j++) w[i][j] = distribution(rnd);
			}
		}
		~Weight()
		{
			if (w != nullptr) for (uint i = 0; i < row; i++) if (w[i] != nullptr) delete[] w[i];
			delete[] w;
		}

		friend void Network::forward(const Weight &weight, const double *const input, const uint &inputlen, double **const output, const uint &outputlen);
		friend void Network::update(const Weight &weight, const double *const delta);
	} *weight;

public:
	enum ActivationType
	{
		Zero,
		One,
		Identity,
		Sigmoid,
		Arctan,
		Tanh,
		Relu,
	};

	Network(const uint &input_size, const uint &hiddenlayercount, const uint &hidden_size, const uint &output_size, ActivationType actType = Zero, const double &alpha = 0.1)
		: inlen(input_size), hiddenlen(hiddenlayercount), hiddensize(new uint[hiddenlayercount - 1]), outlen(output_size), alpha(alpha), 
		weight(new Weight[hiddenlayercount]), input(new double[input_size]), z(new double*[hiddenlayercount]), l(new double*[hiddenlayercount - 1]), output(new double[output_size]),
		activationFunction(new ActFunc*[hiddenlayercount]), dactivationFunction(new ActFunc*[hiddenlayercount])
		// hidden_size represents the number of layers including output but not input.
		// hiddensize[hiddenlayercount - 1] should not be used, use output_size instead.
	{
		weight[0] = Weight(input_size, hidden_size);
		hiddensize[0] = hidden_size;
		for (uint i = 1; i < hiddenlayercount - 1; i++) { weight[i] = Weight(hidden_size, hidden_size); hiddensize[i] = hidden_size; }
		weight[hiddenlayercount - 1] = Weight(hidden_size, output_size);
		hiddensize[hiddenlayercount - 1] = output_size;
		for (uint i = 0; i < hiddenlayercount - 1; i++)
		{
			z[i] = new double[hidden_size];
			l[i] = new double[hidden_size];
			activationFunction[i] = new ActFunc[hidden_size];
			dactivationFunction[i] = new ActFunc[hidden_size];
		}
		z[hiddenlayercount - 1] = new double[output_size];
		activationFunction[hiddenlayercount - 1] = new ActFunc[output_size];
		dactivationFunction[hiddenlayercount - 1] = new ActFunc[output_size];
		switch (actType)
		{
		case Zero:
			for (uint i = 0; i < hiddenlayercount - 1; i++) for (uint j = 0; j < hidden_size; j++) { activationFunction[i][j] = zero; dactivationFunction[i][j] = zero; }
			for (uint i = 0; i < hidden_size; i++) { activationFunction[hiddenlayercount - 1][i] = zero; dactivationFunction[hiddenlayercount - 1][i] = zero; }
			break;
		case One:
			for (uint i = 0; i < hiddenlayercount - 1; i++) for (uint j = 0; j < hidden_size; j++) { activationFunction[i][j] = one; dactivationFunction[i][j] = zero; }
			for (uint i = 0; i < hidden_size; i++) { activationFunction[hiddenlayercount - 1][i] = one; dactivationFunction[hiddenlayercount - 1][i] = zero; }
			break;
		case Identity:
			for (uint i = 0; i < hiddenlayercount - 1; i++) for (uint j = 0; j < hidden_size; j++) { activationFunction[i][j] = identity; dactivationFunction[i][j] = one; }
			for (uint i = 0; i < hidden_size; i++) { activationFunction[hiddenlayercount - 1][i] = identity; dactivationFunction[hiddenlayercount - 1][i] = one; }
			break;
		case Sigmoid:
			for (uint i = 0; i < hiddenlayercount - 1; i++) for (uint j = 0; j < hidden_size; j++) { activationFunction[i][j] = sigmoid; dactivationFunction[i][j] = dsigmoid; }
			for (uint i = 0; i < hidden_size; i++) { activationFunction[hiddenlayercount - 1][i] = sigmoid; dactivationFunction[hiddenlayercount - 1][i] = dsigmoid; }
			break;
		case Arctan:
			for (uint i = 0; i < hiddenlayercount - 1; i++) for (uint j = 0; j < hidden_size; j++) { activationFunction[i][j] = arctan; dactivationFunction[i][j] = darctan; }
			for (uint i = 0; i < hidden_size; i++) { activationFunction[hiddenlayercount - 1][i] = arctan; dactivationFunction[hiddenlayercount - 1][i] = darctan; }
			break;
		case Tanh:
			for (uint i = 0; i < hiddenlayercount - 1; i++) for (uint j = 0; j < hidden_size; j++) { activationFunction[i][j] = tanhyp; dactivationFunction[i][j] = dtanhyp; }
			for (uint i = 0; i < hidden_size; i++) { activationFunction[hiddenlayercount - 1][i] = tanhyp; dactivationFunction[hiddenlayercount - 1][i] = dtanhyp; }
			break;
		case Relu:
			for (uint i = 0; i < hiddenlayercount - 1; i++) for (uint j = 0; j < hidden_size; j++) { activationFunction[i][j] = relu; dactivationFunction[i][j] = drelu; }
			for (uint i = 0; i < hidden_size; i++) { activationFunction[hiddenlayercount - 1][i] = relu; dactivationFunction[hiddenlayercount - 1][i] = drelu; }
			break;
		}
	}
	Network(const uint &input_size, const uint &hiddenlayercount, const uint *hidden_size, const uint &output_size, const ActivationType **const actType, const double &alpha = 0.1)
		: inlen(input_size), hiddenlen(hiddenlayercount), hiddensize(new uint[hiddenlayercount - 1]), outlen(output_size), alpha(alpha),
		weight(new Weight[hiddenlayercount]), input(new double[input_size]), z(new double*[hiddenlayercount]), l(new double*[hiddenlayercount - 1]), output(new double[output_size]),
		activationFunction(new ActFunc*[hiddenlayercount]), dactivationFunction(new ActFunc*[hiddenlayercount])
		// hidden_size represents the number of layers including output but not input.
		// hiddensize[hiddenlayercount - 1] should not be used, use output_size instead.
	{
		weight[0] = Weight(input_size, hidden_size[0]);
		for (uint i = 1; i < hiddenlayercount - 1; i++) { weight[i] = Weight(hidden_size[i - 1], hidden_size[i]); hiddensize[i] = hidden_size[i]; }
		weight[hiddenlayercount - 1] = Weight(hidden_size[hiddenlayercount - 2], output_size);
		hiddensize[hiddenlayercount - 1] = output_size;
		for (uint i = 0; i < hiddenlayercount - 1; i++)
		{
			z[i] = new double[hidden_size[i]];
			l[i] = new double[hidden_size[i]];
			activationFunction[i] = new ActFunc[hidden_size[i]];
			dactivationFunction[i] = new ActFunc[hidden_size[i]];
		}
		z[hiddenlayercount - 1] = new double[output_size];
		activationFunction[hiddenlayercount - 1] = new ActFunc[output_size];
		dactivationFunction[hiddenlayercount - 1] = new ActFunc[output_size];
		for (uint i = 0; i < hiddenlayercount - 1; i++)
		{
			for (uint j = 0; j < hidden_size[i]; j++)
			{
				switch (actType[i][j])
				{
				case Zero:
					activationFunction[i][j] = zero; dactivationFunction[i][j] = zero;
					break;
				case One:
					activationFunction[i][j] = one; dactivationFunction[i][j] = zero;
					break;
				case Identity:
					activationFunction[i][j] = identity; dactivationFunction[i][j] = one;
					break;
				case Sigmoid:
					activationFunction[i][j] = sigmoid; dactivationFunction[i][j] = dsigmoid;
					break;
				case Arctan:
					activationFunction[i][j] = arctan; dactivationFunction[i][j] = darctan;
					break;
				case Tanh:
					activationFunction[i][j] = tanhyp; dactivationFunction[i][j] = dtanhyp;
					break;
				case Relu:
					activationFunction[i][j] = relu; dactivationFunction[i][j] = drelu;
					break;
				}
			}
		}
	}
	~Network()
	{
		if (input != nullptr) delete[] input;
		if (z != nullptr) { for (uint i = 0; i < hiddenlen; i++) { if (z[i] != nullptr) delete[] z[i]; } delete[] z; }
		if (hiddensize != 0) delete[] hiddensize;
		if (output != nullptr) delete[] output;
		if (activationFunction != nullptr) { for (uint i = 0; i < hiddenlen; i++) { if (activationFunction[i] != nullptr) delete[] activationFunction[i]; } delete[] activationFunction; }
		if (dactivationFunction != nullptr) { for (uint i = 0; i < hiddenlen; i++) { if (dactivationFunction[i] != nullptr) delete[] dactivationFunction[i]; } delete[] dactivationFunction; }
		if (weight != nullptr) delete[] weight;
	}

	void ForwardPropagation(const double *const input, const uint &inputlen)
	{
		if (inputlen != inlen) throw std::runtime_error("Error : different length!");

		for (uint i = 0; i < inlen; i++)
		{
			this->input[i] = input[i];
		}
		forward(this->weight[0], input, inputlen, &this->z[0], this->hiddensize[0]);
		for (uint i = 0; i < hiddensize[0]; i++)
		{
			l[0][i] = activationFunction[0][i](this->z[0][i]);
		}

		for (uint i = 1; i < hiddenlen - 1; i++)
		{
			forward(this->weight[i - 1], this->l[i - 1], this->hiddensize[i - 1], &this->z[i], this->hiddensize[i]);
			for (uint j = 0; j < hiddensize[i]; j++)
			{
				l[i][j] = activationFunction[i][j](this->z[i][j]);
			}
		}

		forward(this->weight[hiddenlen - 2], this->l[hiddenlen - 2], hiddensize[hiddenlen - 2], &this->z[hiddenlen - 1], outlen);
		for (uint i = 0; i < outlen; i++)
		{
			output[i] = activationFunction[hiddenlen - 1][i](this->z[hiddenlen - 1][i]);
		}
		/*
		for (uint i = 0; i < outlen; i++)
		{
		std::cout << (output[i] = activationFunction[i](z[i])) << ' ';
		}
		*/
	}
	/// need edit from here
	void BackwardPropagation(const double *const expected, const uint &expectedlen, double &norm)
	{
		if (expectedlen != outlen) throw std::runtime_error("Error : different length!");
		double *delta = new double[outlen]();
		norm = 0;
		for (uint i = 0; i < outlen; i++)
		{
			double x;
			delta[i] = (x = output[i] - expected[i]) * dactivationFunction[i](z[i]);
			norm += x * x;
		}
		update(weight, delta);
		delete[] delta;
	}

	double Learn(const double *const input, const uint &inputlen, const double *const expected, const uint &expectedlen)
	{
		double norm;
		ForwardPropagation(input, inputlen);
		for (uint i = 0; i < outlen; i++) std::cout << '(' << this->output[i] << ',' << ' ' << expected[i] << ')' << ' ';
		std::cout << std::endl;
		BackwardPropagation(expected, expectedlen, norm);
		return norm;
	}
};

inline void Network::forward(const Weight &weight, const double *const input, const uint &inputlen, double **const output, const uint &outputlen)
{
	if (inputlen + 1 != weight.row || outputlen != weight.col) throw std::runtime_error("Error : different length!");

	/// bad way to initialize output, needs optimization

	if ((*output) != nullptr) delete[](*output);
	(*output) = new double[weight.col]();

	for (uint i = 0; i < weight.col; i++) (*output)[i] = weight.w[0][i];
	for (uint i = 1; i < weight.row; i++)
		for (uint j = 0; j < weight.col; j++)
			(*output)[j] += weight.w[i][j] * input[i - 1];
}

inline void Network::update(const Weight &weight, const double *const delta)
{
	for (uint i = 0; i < outlen; i++)
	{
		weight.w[0][i] -= alpha * delta[i];
	}
	for (uint i = 1; i <= inlen; i++)
	{
		for (uint j = 0; j < outlen; j++)
		{
			weight.w[i][j] -= alpha * delta[j] * this->input[i - 1];
		}
	}
}

inline uint pow(uint n)
{
	uint x = 1;
	uint a = 2;
	while (n)
	{
		if (n & 1) x *= a;
		a *= a; n /= 2;
	}
	return x;
}

void int_to_bin(uint n, double *b, uint len)
{
	for (uint i = 0; i < len; i++)
	{
		if (n % 2) b[i] = 1;
		else b[i] = 0;
		n /= 2;
	}
}

int main()
{
	/// input : binary of length n
	/// output : decimal representation of input
	const uint n = 3;
	const uint pn = pow(n);
	Network network(n, n - 1, Network::ActivationType::Identity, 0.4);
	bool b;
	uint loop = 0;
	do
	{
		std::cout << "Loop " << loop++ << std::endl;
		b = true;
		for (uint i = 0; i < pn; i++)
		{
			std::cout << "current(" << i << "): ";
			double *arr = new double[n];
			double *out = new double[n - 1];
			int_to_bin(i, arr, n);
			int tmp = i;
			for (int j = 0; j < n - 1; j++)
			{
				out[j] = tmp / 8.0;
				tmp /= 2;
			}
			b &= (network.Learn(arr, n, out, n - 1) < 0.0001);
			delete[] arr;
			delete[] out;
		}
	} while (!b);
	getchar();
}