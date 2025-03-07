import numpy
import scipy.special


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate

        self.weight_hidden = numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.weight_output = numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs, targets):
        inputs = numpy.array(inputs, ndmin=2).T
        targets = numpy.array(targets, ndmin=2).T

        hidden = numpy.dot(self.weight_hidden, inputs)
        hidden_outputs = self.activation_function(hidden)

        final_inputs = numpy.dot(self.weight_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # output_errors = numpy.array([i ** 2 for i in (targets - final_outputs)]) / len(targets)
        # print(1, output_errors)
        output_errors = targets - final_outputs
        # print(2, output_errors)
        hidden_errors = numpy.dot(self.weight_output.T, output_errors * final_outputs * (1 - final_outputs))
        self.weight_output += self.learning_rate * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.weight_hidden += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

    def query(self, inputs):
        hidden_inputs = numpy.dot(self.weight_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.weight_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("dataset/mnist_train.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

enpochs = 5
for e in range(enpochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        # 不局限于0, 1，小数反而更准确
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("对应的数字为:", correct_number)
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print("推理数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
