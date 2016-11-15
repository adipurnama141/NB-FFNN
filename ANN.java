import java.util.Enumeration;

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
		ANN ann = new ANN();
		BufferedReader reader = new BufferedReader(new FileReader("iris.arff"));
		Instances dataset = new Instances(reader);
		dataset.setClassIndex(dataset.numAttributes() - 1);
		reader.close();

		int numInput = dataset.numAttributes();
		int numOutput = dataset.numClasses();


		System.out.println(dataset);


		System.out.println(dataset.numClasses());
	
		/*
		Enumeration enu = dataset.enumerateInstances();
		while (enu.hasMoreElements()){
			Instance i = (Instance) enu.nextElement();
			System.out.println(i.value(i.attribute(1)));
			System.out.println(i.numAttributes())
		}
		*/


		}
		catch (Exception e){
			e.printStackTrace();
		}

	}

}