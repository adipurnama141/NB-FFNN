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
        private static int classIndex;
        private static double delta;
        private static int numOutput;
        private static int numInput;
        
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
		Instances trainData = new Instances(data , 0, data.numInstances());

		/*Normalize filter = new Normalize();
		filter.setInputFormat(trainData);
		trainData = Filter.useFilter(trainData , filter);*/

		numInput = trainData.numAttributes() - 1;
		numOutput = trainData.numClasses();

	//	System.out.println(numInput);

		
                    Enumeration enu = trainData.enumerateInstances();
                    int countSuccess = 0;
                    while(enu.hasMoreElements()){
                            System.out.println("tes");
                            Instance i = (Instance) enu.nextElement();
                            if (i.classValue() == classifyInstance(i) ){
                                    countSuccess++;
                            }
                    }
                    double successRate = Math.round ((double) countSuccess / (double) trainData.numInstances() * 100);
                    //System.out.println( Math.round(((double) x / 100)) + " : " +successRate + " %");
                    //System.out.println(countSuccess++);
                

	}
        
	public double classifyInstance(Instance instance){
                dataset.setClassIndex(classIndex);
                numOutput = dataset.numClasses();
                int i, j, it, idx;
                double classVal = 0;
                double tempprob =0;
                int numatr= instance.numAttributes();
                int numInst;
                double value;
                ArrayList<double[][]> arr=null;
                boolean found;
                
                //Save all distinct value of class to list
                ArrayList arrcls = new ArrayList();
                for (i = 0; i <dataset.numInstances(); i++) {
                double datumclass= dataset.instance(i).value(classIndex);
                    if (!arrcls.contains(datumclass)) {
                        arrcls.add(datumclass);
                    }
                }
                
                //Classifier
                for (i=0; i<numOutput; i++) {
                    float prob=1;
                    for (j=0; j<numatr; j++) {
                        value = 0;
                        if (j==classIndex) {
                            j++;
                        }
                        arr = discritize(j);
                        numInst = arr.size();
                        found = false;
                        it=0;
                        while (!found && it < numInst ) {
                            value = instance.value(j);
                            if (arr.get(it)[0][0]<= value && arr.get(it)[0][1] > value) {
                                found = true;
                            } else {
                                it++;
                            }
                        }
                        //System.out.println(found);
                        if (found) {
                            prob *= getProbability(j, arr.get(it), (double) arrcls.get(i));
                        } else {
                            return -1;
                        }
                        arr.clear();
                    }
                    prob *= getProbClass(dataset.get(i).value(classIndex));
                    //System.out.println(prob);
                    if (prob>tempprob) {
                        //Bandingkan kelas
                        tempprob = prob;
                        classVal = i;
                    }
                }
		return classVal;
	}
        
        public double getProbClass (double cls) {
            int countcls = 0;
            for (int i=0; i<dataset.numInstances(); i++) {
                if (dataset.get(i).value(classIndex) == cls) {
                    countcls++;
                }
            }
            return (double)countcls/dataset.numInstances();
        }
	public int getFrekuensi(int idxAttrib, double value, double kelas) {
		/*mengembalikan frekuensi idxAttrib yang bernilai kelas (yes/no)*/
		int frek = 0;
		for (int i = 0; i < dataset.numInstances(); i++) {
			if (dataset.instance(i).value(idxAttrib) ==  value &&
				dataset.instance(i).value(classIndex) == kelas) {
				frek++;
			}
		}
		return frek;
	}
        
        public int getFrekuensi(int idxAttrib, double[][] value, double kelas) {
		/*mengembalikan frekuensi idxAttrib yang bernilai kelas (yes/no)*/
		int frek = 0;
                double val;
		for (int i = 0; i < dataset.numInstances(); i++) {
                        val = dataset.instance(i).value(idxAttrib);
			if ( val >=  value[0][0] && val <  value[0][1] &&
				dataset.instance(i).value(classIndex) == kelas) {
				frek++;
			}
		}
		return frek;
	}

	public float getProbability(int idxAttrib, double value, double kelas) {
            ArrayList arr = new ArrayList();
            for (int i=0; i<dataset.numInstances(); i++) {
                double data = dataset.get(i).value(idxAttrib);
                if (!arr.contains(data)) {
                    arr.add(data);
                }
            }
            
            int sumkelas = 0;
            for (int i=0; i<arr.size(); i++) {
                sumkelas += getFrekuensi(idxAttrib, (double) arr.get(i), kelas);
            }
            return (getFrekuensi(idxAttrib, value, kelas)/sumkelas);
	}
        
        public float getProbability(int idxAttrib, double[][] value, double kelas) {
            
                int sumkelas = 0;
                ArrayList<double[][]> arr = discritize(idxAttrib);
                for (int i=0; i<arr.size(); i++) {
                    sumkelas += getFrekuensi(idxAttrib, arr.get(i), kelas);
                }
		return ((float)getFrekuensi(idxAttrib, value, kelas)/sumkelas);
        }
        
        public ArrayList<double[][]> discritize(int idxAttrib) {
            //Get minimum and maksimum value from instance
                double min = dataset.instance(1).value(idxAttrib);
                double maks = min;
                int numInstances = dataset.numInstances();
                for (int it=1; it < numInstances; it++) {
                    double elm = dataset.instance(it).value(idxAttrib);
                    if (elm < min) {
                        min = elm;
                    } else {
                        if (elm > maks) {
                            maks = elm;
                        }
                    }
                }
                //Save all distinct value of instance to list
                delta = 0.2;
                ArrayList<double[][]> arr = new ArrayList<>();
                double start = min;
                while (start<maks+delta) {
                    double[][] datum = new double[1][2];
                    datum[0][0] = start;
                    datum[0][1] = start+delta-0.001;
                    arr.add(datum);
                    start = Math.round((start+delta)*100.00)/100.00;
                }
                
                return arr;
        }
	public void showFrekuensi(int idxAttrib) {
                ArrayList<double[][]> arr = discritize(idxAttrib);
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
                   
                    double[][] elm = arr.get(i);
                    System.out.printf("%.3f-%.3f -> ",elm[0][0],elm[0][1]);
                    for (int j=0; j<arrcls.size(); j++) {
                        double nameclass = (double) arrcls.get(j);
                        int frek = getFrekuensi(idxAttrib, elm, nameclass);
                        System.out.print(""+nameclass+": " +frek+"; ");
                    }
                    System.out.println();
                }
                System.out.println();
	}
        
        public void showProbability(int idxAttrib) {
                ArrayList<double[][]> arr = discritize(idxAttrib);
                System.out.println("--------------");
                
                //Save all distinct value of class to list
                ArrayList arrcls = new ArrayList();
                for (int i = 0; i <dataset.numInstances(); i++) {
                String datumclass= dataset.instance(i).stringValue(classIndex);
                    if (!arrcls.contains(datumclass)) {
                        arrcls.add(datumclass);
                    }
                }
                
                System.out.println(dataset.attribute(idxAttrib).name());
                
                for (int i=0; i<arr.size();i++) {
                   
                    double[][] elm = arr.get(i);
                    System.out.printf("%.3f-%.3f -> ",elm[0][0],elm[0][1]);
                    for (int j=0; j<arrcls.size(); j++) {
                        double nameclass = (double) arrcls.get(j);
                        float prob = getProbability(idxAttrib, elm, nameclass);
                        System.out.print(""+nameclass+": " +prob+"; ");
                    }
                    System.out.println();
                }
                System.out.println();
        }
	public static void main (String args[]) {
		try {
			NB nb = new NB();
			int i, j;
			BufferedReader reader = new BufferedReader(new FileReader("mush.arff"));
			dataset = new Instances(reader);
                        
                        //System.out.println(dataset);
                        classIndex = 0;
			dataset.setClassIndex(classIndex);
			reader.close();
                        /*Evaluation eval = new Evaluation(dataset);
                        System.out.println("tes tes");
                        eval.evaluateModel(nb,dataset);
                        System.out.println("test"+eval.toSummaryString("\nFull Training Results\n", false));*/
                        int numatrinst = dataset.numAttributes();
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
                            
                            //System.out.println("-----------");
                        }
                        System.out.println(((double) counttrue*100)/counttotal);
                        
                        //System.out.println(numatrinst);
                        //Scanner in = new Scanner(System.in);
                        
                        
                        //nb.showFrekuensi(1);
			
		}
		catch (Exception e){
			e.printStackTrace();
		}

	}

}