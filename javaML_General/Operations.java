package javaML_General;

public class Operations {
	
	public double sigmoid(double value) {
		
		return 1.0 / (1.0 + (1.0 / Math.exp(value)));
		
	}
	
	public double sigmoidDerivative(double value) {
		
		return sigmoid(value) * (1.0 - sigmoid(value));
		
	}
	
	public double dot(double[] arr1, double[] arr2) {
		
		double value = 0.0;
		
		for(int i = 0; i < arr1.length; i++) {
			
			value += arr1[i] * arr2[i];
			
		}
		
		return value;
		
	}

}
