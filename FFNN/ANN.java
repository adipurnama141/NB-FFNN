import java.util.Enumeration;
import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;


public class ANN extends AbstractClassifier {
	private Instances 	trainData;

	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.setMinimumNumberInstances(2);
		return result;
	}

	//Proses pembuatan model pembelajaran
	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		data = new Instances(data);
		data.deleteWithMissingClass();
		//copy instancesfrom first (0)
		Instances trainData = new Instances(data , 0, data.numInstances());
	}

	//
	public double classifyInstance(Instance instance){
		/*
		double minDistance = Double.MAX_VALUE;
		double secondMinDistance = Double.MAX_VALUE;
		double distance;
		double classVal = 0;
		double minClassVal = 0;

		Enumeration enu = trainData.enumerateInstances();
		while (enu.hasMoreElements()) {
			Instance trainInstance = (Instancece) enu.nextElement();
			distance = distance(instance, trainInstance);
			if (distance < minDistance) {
				secondMinDistance = minDistance;
				minDistance = distance;

				classVal = minClassVal;
				minClassVal = trainInstance.classValue();
			}
			else if (distance < secondMinDistance){
				secondMinDistance = distance;
				classVal = trainInstance.classValue();
			}
		}
		*/
		double classVal = 0;
		return classVal;
	}

	public static void main (String args[]) {
		try {

		
		BufferedReader reader = new BufferedReader(new FileReader("iris.arff"));
		Instances dataset = new Instances(reader);
		dataset.setClassIndex(dataset.numAttributes() - 1);
		reader.close();

		int numInput = dataset.numAttributes();
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


	


		}
		catch (Exception e){
			e.printStackTrace();
		}

	}

}