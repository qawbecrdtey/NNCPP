// Followed https://www.youtube.com/watch?v=XmK9f5IV8Uw

#include <iostream>

class Neuron
{
private:
	double w;
	double b;
	double input, output;
	double alpha;
public:
	Neuron(const double &w = 1, const double &b = 1, const double &alpha = 0.1) :w(w), b(b), alpha(alpha) {}
	static double ActivationFunction(const double &x)
	{
		// return x > 0 ? x : 0;
		return x;
	}

	static double dActivationFunction(const double &x)
	{
		// return x > 0 ? 1 : (x < 0 ? 0 : 0.5);
		return 1;
	}

	double FeedForward(const double &input)
	{
		std::cout << w << ' ' << b << std::endl;
		return this->output = ((this->input = input) * w + b);
	}

	void BackPropagation(const double &target)
	{
		const double grad = (output - target) * dActivationFunction(output);
		w -= alpha * grad * input;
		b -= alpha * grad;
	}
};

int main()
{
	Neuron neuron(3, 1);
	double x;
	while (abs(4 - (x = neuron.FeedForward(0))) > 0.0001)
	{
		std::cout << x << std::endl;
		neuron.BackPropagation(4);
	}
}