/// copied from one_layer_NN (2018 10 31)
/// needs modification

#include <iostream>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
using uint = unsigned int;
typedef double(*ActFunc)(double);

class Network
{
private:
	class Weight;

	double *input;
	uint inlen;

	double *z;
	double *output;
	uint outlen;

	double alpha;

	ActFunc *activationFunction;
	ActFunc *dactivationFunction;

	static double zero(double x) { return 0; }
	static double one(double x) { return 1; }
	static double identity(double x) { return x; }
	static double sigmoid(double x) { return 1 / (1 + exp(-x)); }
	static double arctan(double x) { return atan(x); }
	static double relu(double x) { return x > 0 ? x : 0; }

	static double dsigmoid(double x) { double t = sigmoid(x); return t * (1 - t); }
	static double darctan(double x) { double t = atan(x); return 1 / (t * t + 1); }
	static double drelu(double x) { return x > 0 ? 1 : (x < 0 ? 0 : 0.5); }

	void forward(const Weight &weight, const double *const input, const uint &inputlen, double **output, const uint &outputlen);
	void update(const Weight &weight, const double *const delta);

	class Weight
	{
	private:
		double **w;
		uint row;
		uint col;

	public:
		Weight(const uint &inlen, const uint &outlen, const double &b = 0)
			:row(inlen + 1), col(outlen), w(new double*[inlen + 1])
		{
			std::default_random_engine generator;
			std::normal_distribution<double> distribution(0, 1);
			w[0] = new double[outlen]();
			for (uint i = 1; i <= inlen; i++)
			{
				w[i] = new double[outlen];
				for (uint j = 0; j < outlen; j++) w[i][j] = distribution(generator);
			}
		}
		~Weight()
		{
			if (w != nullptr)
			{
				for (uint i = 0; i < row; i++)
				{
					if (w[i] != nullptr) delete[] w[i];
				}
			}
			delete[] w;
		}

		friend void Network::forward(const Weight &weight, const double *const input, const uint &inputlen, double **output, const uint &outputlen);
		friend void Network::update(const Weight &weight, const double *const delta);
	} weight;

public:
	enum ActivationType
	{
		Zero,
		One,
		Identity,
		Sigmoid,
		Arctan,
		Relu,
	};

	Network(const uint &input_size, const uint &output_size, ActivationType actType = Zero, const double &alpha = 0.1)
		: inlen(input_size), outlen(output_size), alpha(alpha), weight(Weight(input_size, output_size)),
		input(new double[input_size]), z(new double[output_size]), output(new double[output_size]),
		activationFunction(new ActFunc[output_size]), dactivationFunction(new ActFunc[output_size])
	{
		switch (actType)
		{
		case Zero:
			for (uint i = 0; i < output_size; i++) { activationFunction[i] = zero; dactivationFunction[i] = zero; }
			break;
		case One:
			for (uint i = 0; i < output_size; i++) { activationFunction[i] = one; dactivationFunction[i] = zero; }
			break;
		case Identity:
			for (uint i = 0; i < output_size; i++) { activationFunction[i] = identity; dactivationFunction[i] = one; }
			break;
		case Sigmoid:
			for (uint i = 0; i < output_size; i++) { activationFunction[i] = sigmoid; dactivationFunction[i] = dsigmoid; }
			break;
		case Arctan:
			for (uint i = 0; i < output_size; i++) { activationFunction[i] = arctan; dactivationFunction[i] = darctan; }
			break;
		case Relu:
			for (uint i = 0; i < output_size; i++) { activationFunction[i] = relu; dactivationFunction[i] = drelu; }
			break;
		}
	}
	Network(const uint &input_size, const uint &output_size, const ActivationType *const actType, const double &alpha = 0.1)
		: inlen(input_size), outlen(output_size), alpha(alpha), weight(Weight(input_size, output_size)),
		input(new double[input_size]), z(new double[output_size]), output(new double[output_size]),
		activationFunction(new ActFunc[output_size]), dactivationFunction(new ActFunc[output_size])
	{
		for (uint i = 0; i < output_size; i++)
		{
			switch (actType[i])
			{
			case Zero:
				activationFunction[i] = zero; dactivationFunction[i] = zero;
				break;
			case One:
				activationFunction[i] = one; dactivationFunction[i] = zero;
				break;
			case Identity:
				activationFunction[i] = identity; dactivationFunction[i] = one;
				break;
			case Sigmoid:
				activationFunction[i] = sigmoid; dactivationFunction[i] = dsigmoid;
				break;
			case Arctan:
				activationFunction[i] = arctan; dactivationFunction[i] = darctan;
				break;
			case Relu:
				activationFunction[i] = relu; dactivationFunction[i] = drelu;
				break;
			}
		}
	}
	~Network()
	{
		if (input != nullptr) delete[] input;
		if (z != nullptr) delete[] z;
		if (output != nullptr) delete[] output;
		if (activationFunction != nullptr) delete[] activationFunction;
		if (dactivationFunction != nullptr) delete[] dactivationFunction;
	}

	void ForwardPropagation(const double *const input, const uint &inputlen)
	{
		if (inputlen != inlen) throw std::runtime_error("Error : different length!");

		/// bad way of initializing, needs optimization

		if (this->input != nullptr) delete[] this->input;
		this->input = new double[inlen];
		for (int i = 0; i < inlen; i++)
		{
			this->input[i] = input[i];
		}

		forward(this->weight, input, inputlen, &this->z, outlen);
		for (uint i = 0; i < outlen; i++)
		{
			output[i] = activationFunction[i](this->z[i]);
		}
		/*
		for (uint i = 0; i < outlen; i++)
		{
		std::cout << (output[i] = activationFunction[i](z[i])) << ' ';
		}
		*/
	}
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

inline void Network::forward(const Weight &weight, const double *const input, const uint &inputlen, double **output, const uint &outputlen)
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

///////////////////////////////////////////////////////////////////////////
///
///////////////////////////////////////////////////////////////////////////

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
	const uint n = 2;
	const uint pn = pow(n);
	Network network(n, n - 1, Network::ActivationType::Relu, 0.2);
	bool b;
	uint loop = 0;
	do
	{
		std::cout << "Loop " << loop++ << std::endl;
		b = true;
		for (int i = 0; i < pn; i++)
		{
			std::cout << "current(" << i << "): ";
			double *arr = new double[n];
			double *out = new double[n - 1];
			int_to_bin(i, arr, n);
			int tmp = i;
			for (int j = 0; j < n - 1; j++)
			{
				out[j] = tmp;
				tmp /= 2;
			}
			b &= (network.Learn(arr, n, out, n - 1) < 0.0001);
			delete[] arr;
			delete[] out;
		}
	} while (!b);
	getchar();
}