package javaML_General;

import java.io.IOException;
import java.util.Random;

public class AIO_Network {

	int numInp = 784;
	int numHidsNeu = 128; //Same number as inputs if 0 layers
	int numHidLay = 1;
	int numOut = 10;

	double[] inputs = new double[numInp];
	double[][] hiddenNeurons = new double[numHidLay][numHidsNeu];
	double[] outputs = new double[numOut];

	double[] desiredOuts = new double[outputs.length];

	// WEIGHTS
	double[][] inpTOhid;
	double[][] inpTOhidGRAD;

	double[][][] hidTOhid;
	double[][][] hidTOhidGRAD;

	double[][] hidTOout;
	double[][] hidTOoutGRAD;

	// BIAS
	double[][] biasHid;
	double[][] biasHidGRAD;

	double[] biasOut;
	double[] biasOutGRAD;

	// VALUES
	double step = -0.001;
	double totalBatches = 10000000.0;

	int maxOne = 0;
	double accuracy = 0;
	boolean inTraining = true;

	// Dynamic Step
	boolean dynStepEnable = false;
	double stepLimit = -0.0001;
	double prevAccDS = 0.0;
	double changeThresPERC = -0.1;

	Operations operation = new Operations();

	public AIO_Network() {
		
		// WEIGHTS
		inpTOhid = new double[numHidsNeu][inputs.length];
		inpTOhidGRAD = new double[numHidsNeu][inputs.length];
		
		if(numHidLay != 0) {
			hidTOhid = new double[numHidLay - 1][hiddenNeurons[0].length][hiddenNeurons[0].length];
			hidTOhidGRAD = new double[numHidLay - 1][hiddenNeurons[0].length][hiddenNeurons[0].length];
		}
		
		hidTOout = new double[outputs.length][numHidsNeu];
		hidTOoutGRAD = new double[outputs.length][numHidsNeu];

		// BIAS
		biasHid = new double[numHidLay][numHidsNeu];
		biasHidGRAD = new double[numHidLay][numHidsNeu];

		biasOut = new double[outputs.length];
		biasOutGRAD = new double[outputs.length];
		
		
		Random r = new Random();

		for (int i = 0; i < inpTOhid.length; i++) {

			for (int j = 0; j < inpTOhid[0].length; j++) {

				inpTOhid[i][j] = r.nextDouble();

			}

		}

		for (int i = 0; hiddenNeurons.length != 0 && i < hidTOhid.length; i++) {

			for (int j = 0; j < hidTOhid[0].length; j++) {

				for (int k = 0; k < hidTOhid[0][0].length; k++) {

					hidTOhid[i][j][k] = r.nextDouble();

				}

			}

		}

		for (int i = 0; i < hidTOout.length; i++) {

			for (int j = 0; j < hidTOout[0].length; j++) {

				hidTOout[i][j] = r.nextDouble();

			}

		}

		train();

	}

	public void train() {

		double[][] trainingData = null;
		int[] trainingDataLabel = null;

		MnistReader mr = new MnistReader();
		Random r = new Random();

		try {
			trainingData = mr.readData();
			trainingDataLabel = mr.readDataLabel();

		} catch (IOException e) {

			e.printStackTrace();
			System.err.print("Closing");
			System.exit(1);

		}

		for (int i = 0; i < totalBatches; i++) {

			int item = r.nextInt(trainingData.length);

			desiredOuts = new double[outputs.length];
			desiredOuts[trainingDataLabel[item]] = 1.0;
			inputs = trainingData[item];

			forward();
			BackProp();
			descent();

			hidTOoutGRAD = new double[outputs.length][numHidsNeu];
			inpTOhidGRAD = new double[numHidsNeu][inputs.length];
			
			if(numHidLay != 0)
				hidTOhidGRAD = new double[numHidLay - 1][hiddenNeurons[0].length][hiddenNeurons[0].length];

			biasHidGRAD = new double[numHidLay][numHidsNeu];
			biasOutGRAD = new double[outputs.length];

			if (i % 10000 == 0) {

				System.out.println(
						i + " batches completed, batch accuracy: " + accuracy + " (" + 100 * accuracy / 10000 + "%)");

				if (dynStepEnable)
					dynamicStep();
				
				accuracy = 0;

			}
			
		}

		inTraining = false;
		testSeries();

	}

	public void forward() {
		
		for (int i = 0; hiddenNeurons.length != 0 && i < hiddenNeurons[0].length; i++) {

			hiddenNeurons[0][i] = operation.ReLU(operation.dot(inputs, inpTOhid[i]) + biasHid[0][i]);
			
		}
		
		for (int i = 1; hiddenNeurons.length != 0 && i < hiddenNeurons.length; i++) {

			for (int j = 0; j < hiddenNeurons[0].length; j++) {
				
				hiddenNeurons[i][j] = operation
						.ReLU(operation.dot(hiddenNeurons[i - 1], hidTOhid[i - 1][j]) + biasHid[i][j]);

			}

		}
		
		for (int i = 0; i < outputs.length; i++) {

			if(hiddenNeurons.length != 0)
				outputs[i] = operation
						.sigmoid(operation.dot(hiddenNeurons[hiddenNeurons.length - 1], hidTOout[i]) + biasOut[i]);
			
			else
				outputs[i] = operation.sigmoid(operation.dot(inputs, hidTOout[i]) + biasOut[i]);
			
		}

		// Calculate Accuracy
		maxOne = 0;

		for (int i = 0; i < outputs.length; i++) {

			if (outputs[maxOne] < outputs[i]) {

				maxOne = i;

			}

		}

		if (desiredOuts[maxOne] == 1.0 && inTraining) {

			accuracy += 1.0;

		}

	}

	// BackProp
	public void BackProp() {

		for (int i = 0; i < hidTOout.length; i++) {

			double basis0 = 2.0 * (outputs[i] - desiredOuts[i]);
			biasOutGRAD[i] = basis0;

			for (int j = 0; j < hidTOout[0].length; j++) {
				
				if(hiddenNeurons.length != 0) {
					
					hidTOoutGRAD[i][j] += basis0 * hiddenNeurons[hiddenNeurons.length - 1][j];
					
					double basis1 = basis0 * hidTOout[i][j]
							* operation.ReLUDerivative(hiddenNeurons[hiddenNeurons.length - 1][j]);
					//UnReLU value not required for derivative input; as a 0 ReLU implies 0 derivative
					
					biasHidGRAD[biasHidGRAD.length - 1][j] += basis1;
					
					BackPropPrev(hiddenNeurons.length - 2, j, basis1); //- 2 cuz hidTohid Layer
					
				} else {
					
					hidTOoutGRAD[i][j] += basis0 * inputs[j];
					
				}

			}

		}

	}
	
	// BackProp Chain Rule Handler
	public void BackPropPrev(int layer, int lastNeuron, double basis) {

		if (layer >= 0) {

			for (int i = 0; i < hiddenNeurons[0].length; i++) {

				hidTOhidGRAD[layer][lastNeuron][i] += basis * hiddenNeurons[layer][i];

				double nBasis = basis * hidTOhid[layer][lastNeuron][i]
						* operation.ReLUDerivative(hiddenNeurons[layer][i]);
				biasHidGRAD[layer][i] += nBasis;

				BackPropPrev(layer - 1, i, nBasis);

			}

		} else {

			for (int i = 0; i < inputs.length; i++) {

				inpTOhidGRAD[lastNeuron][i] += basis * inputs[i];

			}

		}

	}

	public void descent() {

		for (int i = 0; i < hidTOout.length; i++) {

			biasOut[i] += step * biasOutGRAD[i];

			for (int j = 0; j < hidTOout[0].length; j++) {

				hidTOout[i][j] += step * hidTOoutGRAD[i][j];

			}

		}

		for (int i = 0; hiddenNeurons.length != 0 && i < hidTOhid.length; i++) {

			for (int j = 0; j < hidTOhid[0].length; j++) {

				biasHid[i + 1][j] += step * biasHidGRAD[i + 1][j];

				for (int k = 0; k < hidTOhid[0][0].length; k++) {

					hidTOhid[i][j][k] += step * hidTOhidGRAD[i][j][k];

				}

			}

		}

		for (int i = 0; hiddenNeurons.length != 0 && i < inpTOhid.length; i++) {

			biasHid[0][i] += step * biasHidGRAD[0][i];

			for (int j = 0; j < inpTOhid[0].length; j++) {

				inpTOhid[i][j] += step * inpTOhidGRAD[i][j];

			}

		}

	}

	public void dynamicStep() {

		double diff = 0.0;

		if (prevAccDS == 0.0) {

			prevAccDS = 100.0 * accuracy / 10000.0;

		} else {

			diff = (100.0 * accuracy / 10000.0) - prevAccDS;

			if (diff < changeThresPERC) {

				if (step < stepLimit) {

					step *= 0.1;
					changeThresPERC *= 0.1;
					System.out.println("New Step Size: " + step);

				}

			}

			prevAccDS = 0.0;

		}

	}

	public void testSeries() {

		accuracy = 1000;

		double[][] trainingData = null;
		int[] trainingDataLabel = null;

		MnistReader mr = new MnistReader();
		Random r = new Random();
		
		try {
			trainingData = mr.readDataTest();
			trainingDataLabel = mr.readDataLabelTest();

		} catch (IOException e) {

			e.printStackTrace();
			System.err.print("Closing");
			System.exit(1);

		}

		for (int i = 0; i < 1000; i++) {

			int item = r.nextInt(trainingData.length);
			inputs = trainingData[item];
			forward();
			if (maxOne != trainingDataLabel[item]) {
				System.out.print("(MISTAKE)");
				accuracy -= 1;
			}

			System.out.println("Predicted: " + maxOne + " Real: " + trainingDataLabel[item] + " Item: " + item);

		}

		System.out.println(accuracy);

	}

}
