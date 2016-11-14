
import java.util.ArrayList;
import java.util.List;
import java.lang.Math;


public class MLPerceptron {

	private List<Perceptron> hiddenLayer = new ArrayList<Perceptron>();
	private List<Perceptron> outputLayer = new ArrayList<Perceptron>();
	private List<Double> desiredOutput = new ArrayList<Double>();

	private int nHiddenLayerNeuron;
	private int nOutputLayerNeuron;
	private int nInput;
	private int nClass;
	private double learningRate;
	private double output;


	public MLPerceptron(int _nInput, int _nHiddenLayerNeuron, int _nClass, double _learningRate ){

		//inisialisasi
		nInput = _nInput;
		nHiddenLayerNeuron = _nHiddenLayerNeuron;
		nClass = _nClass;
		learningRate = _learningRate;

		//inisialisasi hidden layer 
		for (int i = 0 ; i < nHiddenLayerNeuron ; i++){
			hiddenLayer.add(new Perceptron(nInput , learningRate));
		}

		//inisialisasi jumlah neuron 
		if ( nClass == 2){
			nOutputLayerNeuron = 1;
		}
		else {
			nOutputLayerNeuron = nClass;
		}

		//inisialisasi output layer
		for (int i =0 ; i < nOutputLayerNeuron ; i++){
			outputLayer.add(new Perceptron(nHiddenLayerNeuron, learningRate));
		}
	}

	public double getOutput(){
		return output;
	}

	private void determineOutput(){
		double current;
		double maxidx = 0;
		double max = outputLayer.get(0).getOutput();
		for(int i = 0; i < nOutputLayerNeuron ; i++){
			current = outputLayer.get(i).getOutput();
			if (current > max) {
				max = current;
				maxidx = i;
			}
		}

		if (nClass == 2) {
			if (max > 0.5) {
				output = 1;
			}
			else {
				output = 0;
			}
		}
		else {
			output = maxidx;
		}

	}

	public void process(List<Double> in){

		//proses pada hidden layer
		for(int i = 0 ; i < nHiddenLayerNeuron ; i++){
			hiddenLayer.get(i).process(in);
		}

		//buat list hasil proses hidden layer
		List<Double> hiddenLayerResult = new ArrayList<Double>();
		for (int  i = 0; i < nHiddenLayerNeuron ; i++){
			hiddenLayerResult.add(hiddenLayer.get(i).getOutput());
		}

		//proses pada output layer
		for(int i = 0 ; i < nOutputLayerNeuron ; i++){
			outputLayer.get(i).process(hiddenLayerResult);			
		}


		for(int i = 0 ; i < nOutputLayerNeuron ; i++ ){
			System.out.println(outputLayer.get(i).getOutput());
		}

		determineOutput();

	}


	private void prepareDesiredOutput(double d){
		if (nClass == 2){
			desiredOutput.add(new Double(d));			
		}
		else {
			for (double i = 0 ; i < nOutputLayerNeuron ; i = i + 1){
				if (i == d) {
					desiredOutput.add(new Double(1));
				}
				else {
					desiredOutput.add(new Double(0));
				}
			}

		}

		System.out.println("Desired Output : Prepared ");
		for (int i = 0 ; i < desiredOutput.size() ; i++){
			System.out.println(desiredOutput.get(i));
		}
	}


	public void updateWeight(double doutput ){
		double error;
		double perceptronOutput;

		prepareDesiredOutput(doutput);


		//calculate output layer error
		System.out.println("Calculate : Output Layer Error");
		for(int i = 0 ; i <nOutputLayerNeuron ;i++){
			System.out.println("Processing output layer neuron #"+i);
			perceptronOutput = outputLayer.get(i).getOutput();
			System.out.println("Output : "+perceptronOutput);
			error = perceptronOutput * (1 - perceptronOutput) * (desiredOutput.get(i) - perceptronOutput);
			System.out.println("Desired : "+desiredOutput.get(i));
			System.out.println("Error : "+error );
			outputLayer.get(i).setError(error);
		}
		System.out.println("");

		//calculate hidden layer error
		System.out.println("Calculate : Hidden Layer Error");
		for(int i = 0 ; i < nHiddenLayerNeuron ; i++){
			System.out.println("Processing output layer neuron #"+i);

			double sum = 0;
			perceptronOutput = hiddenLayer.get(i).getOutput();
			System.out.println("Output : "+perceptronOutput);

			for (int j = 0 ; j < nOutputLayerNeuron ; j++) {
				double localerror = hiddenLayer.get(j).getError() * hiddenLayer.get(j).getWeight(j+1);
				sum = sum + localerror;
			}

			error = perceptronOutput * ( 1 - perceptronOutput) * sum;
			hiddenLayer.get(i).setError(error);
		}


		//update weight
		for (int i = 0 ; i <nOutputLayerNeuron ; i++){
			outputLayer.get(i).updateWeight();
		}
		for (int i = 0; i <nHiddenLayerNeuron ; i++) {
			hiddenLayer.get(i).updateWeight();
		}
		

	}

	public static void main (String args[]) {
		MLPerceptron mlp = new MLPerceptron(3,5,3,0.9);
		List<Double> input = new ArrayList<Double>();
		input.add(new Double(3.4));
		input.add(new Double(1.3));
		input.add(new Double(0.9));


		for (int i = 0 ; i <100 ; i++){
		mlp.process(input);
		System.out.println(mlp.getOutput());
		mlp.updateWeight(2.0);
		}
		mlp.process(input);
		System.out.println(mlp.getOutput());

	}



}

	


