package ml.examples;

import java.util.Random;
import javax.swing.JFrame;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class ClassificationDemo {

	private Instances data;
	private J48 tree;

	public ClassificationDemo(String arrf) throws Exception {
		//DataSource -> instantiate using WEKA api
		DataSource source = new DataSource(arrf);
		data = source.getDataSet();
		System.out.println(data.numInstances() + " instances loaded!");
		System.out.println(data.toString());
	}

	public void removeFirstAttribute() throws Exception {
		/*
		 * @attribute animal 
		 * {aardvark,antelope,bass,bear,boar,buffalo,calf,carp,catfish
		 * ,cavy,cheetah,chicken,chub,clam,crab,crayfish,crow,deer,dogfish,
		 * dolphin,dove,duck,elephant,flamingo,flea,frog,fruitbat,giraffe,girl...
		 */
		Remove remove = new Remove();
		// remove the first attributes with the instance of String array
		String[] opts = new String[] { "-R", "1" };
		remove.setOptions(opts);
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);
	}

	public void selectFeatures() throws Exception {
		InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
		Ranker ranker = new Ranker();
		AttributeSelection attSelect = new AttributeSelection();
		attSelect.setEvaluator(evaluator);
		attSelect.setSearch(ranker);
		attSelect.SelectAttributes(data);
		int[] selectedAttributes = attSelect.selectedAttributes();
		System.out.println(Utils.arrayToString(selectedAttributes));
	}

	public void buildDecisionTree() throws Exception {
		tree = new J48();
		String[] options = new String[1];
		options[0] = "-U"; // un-pruned tree option
		tree.setOptions(options);
		tree.buildClassifier(data);
		System.out.println(tree.toString());
	}

	public void visualizeTree() throws Exception {
		TreeVisualizer tv = new TreeVisualizer(null, tree.graph(), new PlaceNode2());
		JFrame frame = new JFrame("Tree Visualizer");
		frame.setSize(800, 500);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(tv);
		frame.setVisible(true);
		tv.fitToScreen();
	}
/**
 * Now we have an attribute about that animal,
 * we would like to know what animal is this.
 * It is our classification
 * (Try to put some attributes that represents something).
 * @throws Exception
 */
	public void classifyData() throws Exception {
		double[] vals = new double[data.numAttributes()];
		vals[0] = 1.0;	//hair {false, true}
		vals[1] = 0.0;	//feathers {false, true}
		vals[2] = 0.0;	//eggs {false, true}
		vals[3] = 1.0;	//milk {false, true}
		vals[4] = 0.0;	//airborne {false, true}
		vals[5] = 0.0;	//aquatic {false, true}
		vals[6] = 0.0;	//predator {false, true}
		vals[7] = 1.0;	//toothed {false, true}
		vals[8] = 1.0;	//backbone {false, true}
		vals[9] = 1.0;	//breathes {false, true}
		vals[10] = 1.0;	//venomous {false, true}
		vals[11] = 0.0;	//fins {false, true}
		vals[12] = 4.0;	//legs INTEGER [0,9]
		vals[13] = 1.0;	//tail {false, true}
		vals[14] = 1.0;	//domestic {false, true}
		vals[15] = 0.0;	//catsize {false, true}
		
		Instance myUnicorn = new DenseInstance(1.0, vals);
		myUnicorn.setDataset(data);
		double result = tree.classifyInstance(myUnicorn);
		System.out.println("And the animal is... " + data.classAttribute().value((int) result));
	}

	public void showErrorMetrics() throws Exception {
		Classifier c1 = new J48();
		Evaluation evalRoc = new Evaluation(data);
		evalRoc.crossValidateModel(c1, data, 10, new Random(1), new Object[] {});
		System.out.println(evalRoc.toSummaryString());
		System.out.println(evalRoc.toMatrixString());
	}

	public static void main(String[] args) {
		/** weka.removeFirstAttribute() -> in our examples, we don't need to use all the attributes
		 * need to remove these attributes about our datasets.
		 * 
		 */
		try {
			//instantiate ClassificationDemo to load the dataset. (constructor)
			ClassificationDemo weka = new ClassificationDemo("zoo.arff");
			
			// remove the first attribute.
			weka.removeFirstAttribute();
			
			// what features is the best that describe our classifications?
			// select attributes that have the important features that i need to understand/explain.
			// ans: 12,3,7,2,0,1,8,9,13,4,11,5,15,10,6,14,16
			weka.selectFeatures();
			
			// after feature selection, we will build the decision tree using J48 algorithm.
			weka.buildDecisionTree();
			
			// visualize the tree using Java GUI
			weka.visualizeTree();
			
			// we have datasets that used to build classification (decision) tree,
			// and we have attribute about that animal
			// what if we want to know what animal is this?
			weka.classifyData();
			
			// identify the confusion matrix - important KPI
			// 1. correctly classified instances
			// 2. incorrectly classified instances
			// 3. confusion matrix
			weka.showErrorMetrics();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
