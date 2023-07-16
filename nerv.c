#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// God's in his heaven, all's right with the world.

double S(double x) { return 1 / (1 + exp(-x)); } //Sigmoid function
double dS(double x) { return S(x) * (1 - S(x)); } //Sigmoid function's derivative

double init_weights() { return ((double)rand()) / ((double)RAND_MAX); }

void shuffle(int *array, size_t n) //Shuffles dataset
{
	if (n > 1)
	{
		size_t i;
		for (i = 0; i < n - 1; i++)
		{
			size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
			int t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
	}
}

#define inputs_size 2
#define hidden_nodes_size 2
#define outputs_size 1
#define training_sets_size 4

int main(void)
{
	const double learn_rate = 0.1f;

	double hidden_layer[hidden_nodes_size];
	double output_layer[outputs_size];

	double hidden_layer_biases[hidden_nodes_size];
	double output_layer_biases[outputs_size];

	double hidden_weights[inputs_size][hidden_nodes_size];
	double output_weights[hidden_nodes_size][outputs_size];

	double training_inputs[training_sets_size][inputs_size]
		= { {0.0f, 0.0f},
			{1.0f, 0.0f},
			{0.0f, 1.0f},
			{1.0f, 1.0f} };

	double training_outputs[training_sets_size][outputs_size]
		= { {0.0f},
			{1.0f},
			{1.0f},
			{0.0f} };

	for (int i = 0; i < inputs_size; i++)
	{
		for (int j = 0; j < hidden_nodes_size; j++)		hidden_weights[i][j] = init_weights();
	}

	for (int i = 0; i < hidden_nodes_size; i++)
	{
		hidden_layer_biases[i] = init_weights();
		for (int j = 0; j < outputs_size; j++)			output_weights[i][j] = init_weights();
	}

	for (int i = 0; i < outputs_size; i++)				output_layer_biases[i] = init_weights();

	int training_sets_order[] = { 0, 1, 2, 3 };
	int epochs_count = 10000;

	//Train Nerv for a number of epochs
	for (int epoch = 0; epoch < epochs_count; epoch++)
	{
		shuffle(training_sets_order, training_sets_size);

		for (int x = 0; x < training_sets_size; x++)
		{
			int i = training_sets_order[x];

			/**
			*
			*	FORWARD PASS 
			* 
			**/

			// Compute hidden layer activation
			for (int j = 0; j < hidden_nodes_size; j++)
			{
				double activation = hidden_layer_biases[j];
				for (int k = 0; k < inputs_size; k++)			activation += training_inputs[i][k] * hidden_weights[k][j];
				hidden_layer[j] = S(activation);
			}

			// Compute output layer activation
			for (int j = 0; j < outputs_size; j++)
			{
				double activation = output_layer_biases[j];
				for (int k = 0; k < hidden_nodes_size; k++)		activation += hidden_layer[k] * output_weights[k][j];
				output_layer[j] = S(activation);
			}

			//Print forward pass results
			printf("Input: %g %g	Output: %g	Predicted Output: %g \n",
					training_inputs[i][0], training_inputs[i][1], output_layer[0], training_outputs[i][0]);

			/**
			*
			*	BACKPROPAGATE
			*
			**/

			//Compute change in output weights
			double delta_output[outputs_size];
			for (int j = 0; j < outputs_size; j++)
			{
				double error = training_outputs[i][j] - output_layer[j];
				delta_output[j] = error * dS(output_layer[j]);
			}

			//Compute change in hidden weights
			double delta_hidden[hidden_nodes_size];
			for (int j = 0; j < hidden_nodes_size; j++)
			{
				double error = 0.0f;
				for (int k = 0; k < outputs_size; k++)			error += delta_output[k] * output_weights[j][k];
				delta_hidden[j] = error * dS(hidden_layer[j]);
			}

			//Apply change in output weights
			for (int j = 0; j < outputs_size; j++)
			{
				output_layer_biases[j] += delta_output[j] * learn_rate;
				for (int k = 0; k < hidden_nodes_size; k++)		output_weights[k][j] += hidden_layer[k] * delta_output[j] * learn_rate;
			}

			//Apply change in hidden weights
			for (int j = 0; j < hidden_nodes_size; j++)
			{
				hidden_layer_biases[j] += delta_hidden[j] * learn_rate;
				for (int k = 0; k < inputs_size; k++)			hidden_weights[k][j] += training_inputs[i][k] * delta_hidden[j] * learn_rate;
			}

		}
	}

	//Print final weights after training
	fputs("\nFinal Hidden Weights\n[\n", stdout);
	for (int j = 0; j < hidden_nodes_size; j++)
	{
		fputs("[ ", stdout);
		for (int k = 0; k < inputs_size; k++)			printf("%f ", hidden_weights[k][j]);
		fputs(" ]\n", stdout);
	}

	fputs("]\nFinal Hidden Biases\n[ ", stdout);
	for (int j = 0; j < hidden_nodes_size; j++)			printf("%f ", hidden_layer_biases[j]);

	fputs("]\nFinal Output Weights\n[\n", stdout);
	for (int j = 0; j < outputs_size; j++)
	{
		fputs("[ ", stdout);
		for (int k = 0; k < hidden_nodes_size; k++)		printf("%f ", output_weights[k][j]);
		fputs(" ]\n", stdout);
	}

	fputs("]\nFinal Output Biases\n[ ", stdout);
	for (int j = 0; j < outputs_size; j++)				printf("%f ", output_layer_biases[j]);
	fputs(" ] \n", stdout);

	return 0;
}