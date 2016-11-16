import java.util.Scanner;
import java.util.Enumeration;
import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;


import java.io.BufferedReader;
import java.io.FileReader;


public class ANN extends AbstractClassifier {
	private Instances trainData;
	private MLPerceptron mlp;
	private int numInput;
	private int numOutput;
	private double learningRate;
	private int numHiddenLayerNeuron;

	public ANN(double _learningRate , int _numHiddenLayerNeuron ) {
		learningRate = _learningRate;
		numHiddenLayerNeuron = _numHiddenLayerNeuron;
	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.setMinimumNumberInstances(2);
		return result;
	}

	//Proses pembuatan model pembelajaran
	@Override
	public void buildClassifier(Instances data) throws Exception {
		data = new Instances(data);
		data.deleteWithMissingClass();
		Instances trainData = new Instances(data , 0, data.numInstances());

		Normalize filter = new Normalize();
		filter.setInputFormat(trainData);
		trainData = Filter.useFilter(trainData , filter);


		numInput = trainData.numAttributes() - 1;
		numOutput = trainData.numClasses();

		System.out.println(numInput);

		mlp = new MLPerceptron(numInput,numHiddenLayerNeuron,numOutput,learningRate);

		List<Double> input = new ArrayList<Double>();
		

		for (int x = 0 ; x < 10000 ; x++){
		Enumeration enu = trainData.enumerateInstances();
		int countSuccess = 0;
		while(enu.hasMoreElements()){
			Instance i = (Instance) enu.nextElement();
			for (int j = 0 ; j<numInput ; j++){
				input.add(new Double(i.value(i.attribute(j))));
			}
			mlp.process(input);
			if (i.classValue() == mlp.getOutput() ){
				countSuccess++;
			}
			mlp.updateWeight(i.classValue());
			input.clear();
		}
		double successRate = Math.round ((double) countSuccess / (double) trainData.numInstances() * 100);
		//System.out.println( Math.round(((double) x / 100)) + " : " +successRate + " %");
		//System.out.println(countSuccess++);
	}

	}

	public double classifyInstance(Instance instance){
			List<Double> input = new ArrayList<Double>();
			Instance i = instance;

			//Construct input
			System.out.println("Instance!");
			for (int j = 0 ; j<numInput ; j++){
				input.add(new Double(i.value(i.attribute(j))));
			}
			//process
			mlp.process(input);

		return mlp.getOutput();
	}

	public static void main (String args[]) {
		try {

		
		BufferedReader reader = new BufferedReader(new FileReader("iris.arff"));
		Instances dataset = new Instances(reader);
		dataset.setClassIndex(dataset.numAttributes() - 1);
		reader.close();

		Scanner scan = new Scanner(System.in);
		System.out.print("Masukkan jumlah neuron pada hidden layer : ");
		int n = scan.nextInt();


		ANN ann = new ANN(0.2  , n	);
		ann.buildClassifier(dataset);


		System.out.println("BATAS!");

		Normalize filter = new Normalize();
		filter.setInputFormat(dataset);
		dataset = Filter.useFilter(dataset , filter);

		Evaluation eval = new Evaluation(dataset);
		eval.evaluateModel(ann,dataset);
		System.out.println(eval.toSummaryString("\nFull Training Results\n", false));


		/*int numInput = dataset.numAttributes();
		int numOutput = dataset.numClasses();

		MLPerceptron mlp = new MLPerceptron(numInput,5,numOutput,0.1);
		List<Double> input = new ArrayList<Double>();


		for (int x = 0 ; x < 100 ; x++){
		Enumeration enu = dataset.enumerateInstances();
		while(enu.hasMoreElements()){
			Instance i = (Instance) enu.nextElement();

			//Construct input
			System.out.println("Instance!");
			for (int j = 0 ; j<numInput ; j++){
				//System.out.println(i.value(i.attribute(j)));
				input.add(new Double(i.value(i.attribute(j))));
			}
			//process
			mlp.process(input);

			//learn
			System.out.println(i.classValue());
			mlp.updateWeight(i.classValue());
			input.clear();
		}
		

	}
	*/

	


		}
		catch (Exception e){
			e.printStackTrace();
		}

	}

}