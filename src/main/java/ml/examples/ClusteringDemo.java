package ml.examples;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Random;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;

public class ClusteringDemo {

	private Instances data;

	public ClusteringDemo(String file) throws IOException {
		InputStream inputStream = ClusteringDemo.class.getResourceAsStream("/" + file);
		data = new Instances(new BufferedReader(new InputStreamReader(inputStream)));
		System.out.println(data);
	}

	public void buildCluster() throws Exception {
		EM model = new EM();
		model.buildClusterer(data);
		System.out.println(model);
	}

	public void evaluationModel() throws Exception {
		double crossValidateModel = ClusterEvaluation.crossValidateModel(new EM(), data, 10, new Random(1));
		System.out.println(crossValidateModel);
	}

	public static void main(String[] args) {
		try {
			ClusteringDemo cluster = new ClusteringDemo("bank.arff");
			cluster.buildCluster();
			cluster.evaluationModel();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
