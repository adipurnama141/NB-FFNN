/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

/**
 *
 * @author Ikhwan
 */
import java.util.Enumeration;

import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Scanner;
import weka.classifiers.Evaluation;
import weka.core.DenseInstance;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;


public class NB extends AbstractClassifier {
        private static Instances dataset;
        private static int classIndex=0;
        private static double delta=1;
        private static int numOutput;
        private static int numInput;
        private static ArrayList<ArrayList<ArrayList<Float>>> peluang;
        private static ArrayList<ArrayList<ArrayList<Double>>> probs;
        
        
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.setMinimumNumberInstances(2);
		return result;
	}

	//Proses pembuatan model pembelajaran
        public void printProb() {
            for (int i=0; i<dataset.numAttributes(); i++) {
                ArrayList<ArrayList<Double>> tes = probs.get(i);
                for (int j=0; j< tes.size(); j++) {
                    System.out.println(j-1);
                    System.out.println(tes.get(j));
                }
            }
        }
	@Override
	public void buildClassifier(Instances data) throws Exception {
            probs = new ArrayList<>();
            Instances dt = new Instances(data);
            for (int i=0; i<dt.numAttributes(); i++) {
                if (i!=classIndex) {
                    probs.add(i, getProbabilityAttribute(i, dt));
                } else {
                    probs.add(i, getProbabilityClass(dt));
                }
            }
	}
        
        @Override
	public double classifyInstance(Instance instance){
                dataset.setClassIndex(classIndex);
                numOutput = dataset.numClasses();
                int i, atr, it;
                double classVal = 0;
                double tempprob =0;
                int numatr= instance.numAttributes();
                int numInst;
                double value;
                boolean found;
                
                //Classifier
                for (i=0; i<numOutput; i++) {
                    double prob=1;
                    for (atr=0; atr<numatr; atr++) {
                        ArrayList<Double> arr = new ArrayList<Double>();
                        if (atr==classIndex) {
                            atr++;
                        }
                        arr = probs.get(atr).get(0);
                        numInst = arr.size();
                        found = false;
                        it=0;
                        while (!found && it < numInst ) {
                            value = instance.value(atr);
                            if (arr.get(it) <= value && arr.get(it)+delta-0.001 > value) {
                                found = true;
                                
                            } else {
                                it++;
                            }
                        }
                        if (found) {
                            prob *= probs.get(atr).get(i+1).get(it);
                        } else {
                            return -1;
                        }
                    }
                    //System.out.println(probs.get(classIndex).get(1).get(i));
                    prob *= probs.get(classIndex).get(1).get(i);
                    //System.out.println(prob);
                    if (prob>tempprob) {
                        //Bandingkan kelas
                        tempprob = prob;
                        classVal = (double) i;
                    }
                }
		return classVal;
	}
        
        public double getProbabilityIdxClass (double cls, Instances data) {
            int countcls = 0;
            for (int i=0; i<data.numInstances(); i++) {
                if (data.get(i).value(classIndex) == cls) {
                    countcls++;
                }
            }
            return (double)countcls/dataset.numInstances();
        }
        
        public ArrayList<ArrayList<Double>> getProbabilityClass (Instances data) {
            //Save all distinct value of class to list
            ArrayList<Double> arrcls = new ArrayList();
            for (int i = 0; i <data.numInstances(); i++) {
            double datumclass= data.instance(i).value(classIndex);
                if (!arrcls.contains(datumclass)) {
                    arrcls.add(datumclass);
                }
            }
            ArrayList<ArrayList<Double>> probclass = new ArrayList<>();
            probclass.add(arrcls);
            ArrayList<Double> temp = new ArrayList<>();
            for (int j=0; j<arrcls.size(); j++) {
                temp.add(getProbabilityIdxClass((double) j, data));
            }
            probclass.add(temp);
            return probclass;
        }
	public int getFrekuensi(int idxAttrib, double value, double kelas, Instances data) {
		/*mengembalikan frekuensi idxAttrib yang bernilai kelas (yes/no)*/
		int frek = 0;
                double val;
		for (int i = 0; i < data.numInstances(); i++) {
			val = data.instance(i).value(idxAttrib);
			if ( val >=  value && val <  value+delta-0.001 &&
				data.instance(i).value(classIndex) == kelas) {
				frek++;
			}
		}
		return frek;
	}
        
       
	public float getProbability(int idxAttrib, double value, double kelas, Instances data) {
            int sumkelas = 0;
                ArrayList<Double> arr = discritize(idxAttrib, data);
                for (int i=0; i<arr.size(); i++) {
                    sumkelas += getFrekuensi(idxAttrib, arr.get(i), kelas, data);
                }
		return ((float)getFrekuensi(idxAttrib, value, kelas, data)/sumkelas);
	}
        
        
        public ArrayList discritize(int idxAttrib, Instances data) {
            //Get minimum and maksimum value from instance
                double min = data.instance(1).value(idxAttrib);
                double maks = min;
                int numInstances = data.numInstances();
                for (int it=1; it < numInstances; it++) {
                    double elm = data.instance(it).value(idxAttrib);
                    if (elm < min) {
                        min = elm;
                    } else {
                        if (elm > maks) {
                            maks = elm;
                        }
                    }
                }
                //Save all distinct value of instance to list
                ArrayList arr = new ArrayList();
                double start = min;
                while (start<maks+delta) {
                    double datum = start;
                    arr.add(datum);
                    start = Math.round((start+delta)*100.00)/100.00;
                }
                
                return arr;
        }
	
        
        public void showProbability(int idxAttrib, Instances data) {
                ArrayList<Double> arr = discritize(idxAttrib, dataset);
                System.out.println("--------------");
                
                //Save all distinct value of class to list
                ArrayList arrcls = new ArrayList();
                for (int i = 0; i <dataset.numInstances(); i++) {
                double datumclass= dataset.instance(i).value(classIndex);
                    if (!arrcls.contains(datumclass)) {
                        arrcls.add(datumclass);
                    }
                }
                
                System.out.println(dataset.attribute(idxAttrib).name());
                
                for (int i=0; i<arr.size();i++) {
                   
                    double elm = arr.get(i);
                    System.out.printf("%.3f-%.3f -> ",elm,elm+delta-0.001);
                    for (int j=0; j<arrcls.size(); j++) {
                        double nameclass = (double) arrcls.get(j);
                        float prob = getProbability(idxAttrib, elm, nameclass, data);
                        System.out.print(""+nameclass+": " +prob+"; ");
                    }
                    System.out.println();
                }
                System.out.println();
        }
        
        public ArrayList<ArrayList<Double>> getProbabilityAttribute(int idxAttrib, Instances data) {
                ArrayList<ArrayList<Double>> probArr = new ArrayList<>();
                ArrayList<Double> arr = discritize(idxAttrib, data);
                probArr.add(arr);
                //Save all distinct value of class to list
                ArrayList arrcls = new ArrayList();
                for (int i = 0; i <data.numInstances(); i++) {
                double datumclass= data.instance(i).value(classIndex);
                    if (!arrcls.contains(datumclass)) {
                        arrcls.add(datumclass);
                    }
                }
                
                //System.out.println(dataset.attribute(idxAttrib).name());
                for (int j=0; j<arrcls.size(); j++) {
                    ArrayList<Double> arrClass = new ArrayList<>();
                    double nameclass = (double) arrcls.get(j);
                    for (int i=0; i<arr.size(); i++) {
                        double elm = arr.get(i);
                        double prob = getProbability(idxAttrib, elm, nameclass, data);
                        arrClass.add(prob);
                    }
                    probArr.add(arrClass);
                }
                return probArr;
        }
	public static void main (String args[]) {
		try {
			NB nb = new NB();
			int i, j;
			BufferedReader reader = new BufferedReader(new FileReader("mush.arff"));
			dataset = new Instances(reader);
                        
                        classIndex = 0;
			dataset.setClassIndex(classIndex);
			reader.close();
                        
                        //BUILD CLASSIFIER
                        nb.buildClassifier(dataset);
                        Evaluation eval = new Evaluation(dataset);
                        eval.evaluateModel(nb, dataset);
                        System.out.println(eval.toSummaryString());
                        System.out.println(eval.toMatrixString());
                        /*int numatrinst = dataset.numAttributes();
                        //Input instance baru
                        Instance inst = new DenseInstance(numatrinst);
                        BufferedReader in = new BufferedReader(new FileReader("tes.txt"));
                        String line = null;
                        
                        //Save all distinct value of class to list
                        ArrayList arrcls = new ArrayList();
                        for (i = 0; i <dataset.numInstances(); i++) {
                        String datumclass= dataset.instance(i).stringValue(classIndex);
                            if (!arrcls.contains(datumclass)) {
                                arrcls.add(datumclass);
                            }
                        }
                        int numInst = dataset.numInstances();
                        //process
                        int counttrue =0;
                        int counttotal = 0;
                        int missingclass = 0;
                        while ((line = in.readLine())!=null) {
                            counttotal++;
                            String[] _line = line.split(" ");
                            for (i=0; i<numatrinst; i++) {
                                if (i==classIndex) {
                                    i++;
                                }
                                int k=0;
                                boolean found = false;
                                
                                while (!found && k < numInst) {
                                    if (dataset.instance(k).stringValue(i).equals(_line[i])) {
                                        found = true;
                                    } else {
                                        k++;
                                    }
                                }
                                if (found) {
                                    inst.setValue(i, dataset.instance(k).value(i));
                                } else {
                                    //System.out.println("Tidak bisa mengklasifikasi.");
                                    break;
                                }
                            }
                            
                            //System.out.println("Instance : "+inst);
                            double classVal = nb.classifyInstance(inst);
                            if (classVal >= 0) {
                                //System.out.println("Klasifikasi : "+arrcls.get((int) classVal));
                                if (arrcls.get((int) classVal).equals(_line[0])) {
                                    counttrue++;
                                }
                            } else {
                                //System.out.println("Tidak dapat mengklasifikasi.");
                            }
                        }
                        System.out.println("Total : "+counttotal);
                        System.out.println("Tepat : "+counttrue);
                        System.out.println("Miss : "+(counttotal-counttrue));
                        System.out.println("Akurasi : "+((double) counttrue*100)/counttotal +"%");*/
                        
		}
		catch (Exception e){
			e.printStackTrace();
		}

	}

}