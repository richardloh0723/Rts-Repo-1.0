package ml.examples;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class LinearRegressionDemo {

	private Instances data;

	public LinearRegressionDemo(String csv) throws IOException {
		CSVLoader csvLoader = new CSVLoader();
		csvLoader.setSource(new File(csv));
		data = csvLoader.getDataSet();
//		System.out.println(data);
	}

	public void buildLinearRegression() throws Exception {
		data.setClassIndex(data.numAttributes() - 2);

		Remove remove = new Remove();
		remove.setOptions(new String[] { "-R", data.numAttributes() + "" });
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);
		
		System.out.println(data);
		
		LinearRegression model = new LinearRegression();
		model.buildClassifier(data);
		System.out.println(model);
	}

	public void evaluationModel() throws Exception {
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(new LinearRegression(), data, 10, new Random(1), new String[] {});
		System.out.println(eval.toSummaryString());
	}

	public static void main(String[] args) {
		try {
			LinearRegressionDemo regression = new LinearRegressionDemo("ENB2012_data.csv");
			regression.buildLinearRegression();
			regression.evaluationModel();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
