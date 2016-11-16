
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
	private boolean noHiddenLayer;


	public MLPerceptron(int _nInput, int _nHiddenLayerNeuron, int _nClass, double _learningRate ){

		//inisialisasi
		nInput = _nInput;
		nHiddenLayerNeuron = _nHiddenLayerNeuron;
		nClass = _nClass;
		learningRate = _learningRate;

		if ( nHiddenLayerNeuron == 0 ){
			noHiddenLayer = true;
			nHiddenLayerNeuron = nInput;
		}
		else {
			noHiddenLayer = false;
			
		}

		if (!noHiddenLayer){
			//inisialisasi hidden layer 
			for (int i = 0 ; i < nHiddenLayerNeuron ; i++){
				hiddenLayer.add(new Perceptron(nInput , learningRate));
			}
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
			//System.out.println("n Hidden layer" + nHiddenLayerNeuron);
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

		List<Double> hiddenLayerResult;
		//System.out.println("FEED FORWARD ANN (BEGIN)");
		//System.out.println("Input : "  + in);
		//System.out.println("-----------------------------------------------------------");
		//System.out.println("");
		

		//System.out.println("PROPAGATE");

		
		if (!noHiddenLayer){
			//proses pada hidden layer
			for(int i = 0 ; i < nHiddenLayerNeuron ; i++){
				//System.out.println("HiddenLayer Neuron #" + i);
				hiddenLayer.get(i).process(in);
			}

			//buat list hasil proses hidden layer
			hiddenLayerResult= new ArrayList<Double>();
			for (int  i = 0; i < nHiddenLayerNeuron ; i++){
				hiddenLayerResult.add(hiddenLayer.get(i).getOutput());
			}
		}
		else {
			hiddenLayerResult = in;
		}


		//System.out.println("Processing Hidden Layer  : " + hiddenLayerResult);
		//System.out.println("");

		//proses pada output layer
		for(int i = 0 ; i < nOutputLayerNeuron ; i++){
			//System.out.println("OutputLayer Neuron #"+i);
			outputLayer.get(i).process(hiddenLayerResult);			
		}

		determineOutput();

		//System.out.println("Processing Output Layer  : ");
		for(int i = 0 ; i <nOutputLayerNeuron ; i++){
			//System.out.println(outputLayer.get(i).getOutput());
		}

		//System.out.println("Output : " + output);

	}


	private void prepareDesiredOutput(double d){
		desiredOutput.clear();
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

	}


	public void updateWeight(double doutput ){
		double error;
		double perceptronOutput;

		//System.out.println("Real Desired Output : " + doutput);
		//System.out.println("");
		prepareDesiredOutput(doutput);
		//System.out.println(desiredOutput);

		//System.out.println("");
		//System.out.println("BACKPROPAGATE");
		//System.out.println("");


		//calculate output layer error

		for(int i = 0 ; i <nOutputLayerNeuron ;i++){
			perceptronOutput = outputLayer.get(i).getOutput();
			error = perceptronOutput * (1 - perceptronOutput) * (desiredOutput.get(i) - perceptronOutput);
			outputLayer.get(i).setError(error);

			//System.out.println("Output Layer Neuron#"+i);
			//System.out.println("Desired Output : " + desiredOutput.get(i));
			//System.out.println("Actual Output :" + perceptronOutput);
			//System.out.println("Error  : "+error);
			//System.out.println("");
		}
		

		if (!noHiddenLayer){
			//calculate hidden layer error
			for(int i = 0 ; i < nHiddenLayerNeuron ; i++){
				double sum = 0;
				perceptronOutput = hiddenLayer.get(i).getOutput();
				//System.out.println("Hidden Layer Neuron#"+i);

				for (int j = 0 ; j < nOutputLayerNeuron ; j++) {
					//System.out.print("Output layer # "+ j  + " : ");
					//System.out.println(outputLayer.get(j).getError() + " x " +outputLayer.get(j).getWeight(i+1));
					double localerror = outputLayer.get(j).getError() * outputLayer.get(j).getWeight(i+1);
					sum = sum + localerror;
				}

				error = perceptronOutput * ( 1 - perceptronOutput) * sum;
				hiddenLayer.get(i).setError(error);
				
				//System.out.println("Error  : "+error);
				//System.out.println("");
			}
		}

		//System.out.println("WEIGHT UPDATE");

		//update weight
		for (int i = 0 ; i <nOutputLayerNeuron ; i++){
			//System.out.println("Update Output Layer #"+i);
			outputLayer.get(i).updateWeight();
			//System.out.println("");
		}
		if (!noHiddenLayer){
		for (int i = 0; i <nHiddenLayerNeuron ; i++) {
			//System.out.println("Update Hidden Layer #"+i);
			hiddenLayer.get(i).updateWeight();
			//System.out.println("");
		}
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
			mlp.updateWeight(2.0);
		}
		mlp.process(input);
		

	}



}

	


