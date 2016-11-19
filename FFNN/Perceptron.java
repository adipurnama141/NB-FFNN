import java.util.ArrayList;
import java.util.List;
import java.lang.Math;
import java.util.Random;
import java.io.Serializable;

public class Perceptron implements Serializable{
	
	private int nInput;
	private List<Double> inputs = new ArrayList<Double>();
	private List<Double> weights = new ArrayList<Double>();
	private List<Double> weightedInputs = new ArrayList<Double>();
	private Double summed;
	private Double output;
	private Double learningRate;
	private double error;

	private static Double sigmoid(Double x){
		return (1/(1 + Math.pow(Math.E,(-1*x))));
	}

	public void showWeights(){
		//System.out.println("Current Weights : " + weights.size());
		for (int i = 0; i < weights.size() ; i++){
			//System.out.println(weights.get(i));
		}
	}

	public Double getOutput(){
		return output;
	}



	public Double process(List<Double> in){

		//tambahkan bias
		this.inputs = new ArrayList<Double>(in);
		inputs.add(0, new Double(1));

		//System.out.println("Inputs w Bias : " + inputs);

		//System.out.println("Weight : "+weights);

		//weight input
		for(int i=0 ; i < inputs.size() ; i++){
			weightedInputs.set(i, weights.get(i) * inputs.get(i) );
		}

		//System.out.println("Weighted Inputs : " + weightedInputs);

		//summer
		summed = new Double(0);
		for (int i=0 ; i < weightedInputs.size() ; i++) {
			summed = summed + weightedInputs.get(i);
		}

		//System.out.println("Sum :" + summed);

		//activation function
		output = sigmoid(summed);

		//System.out.println("Sigmoid : " + output);
		//System.out.println("");
		return output;
	}

	public Double getWeight(int idx){
		return weights.get(idx);
	}


	public void setError(double e){
		error = e;
	}

	public double getError(){
		return error;
	}


	public void updateWeight(){
		//System.out.println("Error : " + error);
		double oldweight;
		double newWeight;	
		for(int i=0 ; i< weights.size() ; i++){
			oldweight = weights.get(i);
			newWeight = weights.get(i) + (learningRate * error * inputs.get(i));
			weights.set(i ,newWeight );

			//System.out.println(oldweight +"->"+ newWeight);
		}
	}


	public void updateWeight(double desiredOutput){
		double newWeight;
		error = desiredOutput - output;	
		for(int i=0 ; i< weights.size() ; i++){
			newWeight = weights.get(i) + (learningRate * error * inputs.get(i));
			weights.set(i ,newWeight );
		}
	}

	public Perceptron(int nInput , double learningRate){
		this.nInput = nInput;
		this.learningRate = new Double(learningRate);

		Random rnd = new Random();
		for (int i = 0 ; i <= nInput ; i++){
			//
			weights.add(rnd.nextDouble());
			weightedInputs.add(new Double(0));
		}
	}

	public static void main(String args[]){
		Perceptron p = new Perceptron(3,0.9);
		List<Double> input = new ArrayList<Double>();
		input.add(new Double(3.4));
		input.add(new Double(1.3));
		input.add(new Double(0.9));

		for(int i = 0 ; i < 100 ; i++){
			//System.out.println(p.process(input));
			//p.showWeights();
			p.updateWeight(1);
		}

		//System.out.println(p.process(input));


	}

}