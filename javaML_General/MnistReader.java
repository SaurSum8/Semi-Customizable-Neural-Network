package javaML_General;

import java.io.*;

public class MnistReader {
	
	public double[][] readData() throws IOException {
		
		DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(
				"D:\\Eclipse IDE\\workspace\\First_ML\\train-images.idx3-ubyte")));
		
        dataInputStream.readInt(); //Magic Number
        int totalItems = dataInputStream.readInt();
        int rows = dataInputStream.readInt();
        int cols = dataInputStream.readInt();
        
        double[][] data = new double[totalItems][cols * rows];
        
        for(int i = 0; i < totalItems; i++) {
        	
        	for(int j = 0; j < cols * rows; j++) {
        		
        		data[i][j] = (double) ((double) dataInputStream.readUnsignedByte() / (double) 255.0); //255 is largest possible value
        		
        	}
        	
        }
        
        dataInputStream.close();
        
        return data;
        
	}

	public int[] readDataLabel() throws IOException {
		
		DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(
				"D:\\Eclipse IDE\\workspace\\First_ML\\train-labels.idx1-ubyte")));
        
        labelInputStream.readInt(); //Magic Number
        int totalLabels = labelInputStream.readInt();
        
        int[] dataLabel = new int[totalLabels];
        
        for(int i = 0; i < totalLabels; i++) {
        	
        	dataLabel[i] = labelInputStream.readUnsignedByte();
        	
        }
        
        labelInputStream.close();
		
        return dataLabel;
        
	}
	
	public double[][] readDataTest() throws IOException {
		
		DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(
				"D:\\Eclipse IDE\\workspace\\First_ML\\t10k-images.idx3-ubyte")));
		
        dataInputStream.readInt(); //Magic Number
        int totalItems = dataInputStream.readInt();
        int rows = dataInputStream.readInt();
        int cols = dataInputStream.readInt();
        
        double[][] data = new double[totalItems][cols * rows];
        
        for(int i = 0; i < totalItems; i++) {
        	
        	for(int j = 0; j < cols * rows; j++) {
        		
        		data[i][j] = (double) ((double) dataInputStream.readUnsignedByte() / (double) 255.0); //255 is largest possible value
        		
        	}
        	
        }
        
        dataInputStream.close();
        
        return data;
        
	}

	public int[] readDataLabelTest() throws IOException {
		
		DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(
				"D:\\Eclipse IDE\\workspace\\First_ML\\t10k-labels.idx1-ubyte")));
        
        labelInputStream.readInt(); //Magic Number
        int totalLabels = labelInputStream.readInt();
        
        int[] dataLabel = new int[totalLabels];
        
        for(int i = 0; i < totalLabels; i++) {
        	
        	dataLabel[i] = labelInputStream.readUnsignedByte();
        	
        }
        
        labelInputStream.close();
		
        return dataLabel;
        
	}
	
}