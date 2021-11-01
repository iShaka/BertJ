import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.nlp.qa.*;
import ai.djl.repository.zoo.*;
import ai.djl.training.util.*;
import ai.djl.translate.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

public class TestBertRelationClassification {

    public void calculatePerformance(List<String> gold, List<String> prediction,String neg){
//        assert gold.size()==prediction.size();
        gold=gold.subList(0,prediction.size());
        for(int i=0;i<gold.size();i++){
            System.out.println(gold.get(i)+"\t"+prediction.get(i));
        }
        double tp=0;
        double allPred=0;
        double allGold=0;
        for(int i=0;i<gold.size();i++){
            String g=gold.get(i);
            String p=prediction.get(i);
            if(!g.equals(neg)){allGold++;}
            if(!p.equals(neg)){allPred++;}
            if(g.equals(p)&&(!g.equals(neg))){
                tp++;
            }
        }
        double p=0;
        double r=0;
        double f=0;
        p= tp/allPred;
        r=tp/allGold;
        f=2*p*r/(p+r);
        System.out.println(tp+" "+allPred+" "+allGold);
        System.out.println(p+" "+r+" "+f);
    }

    List<String> gold=new ArrayList<>();
    List<String> sentences =new ArrayList<>();
    List<String> prediction =new ArrayList<>();

    public void readData(String dataFile) throws IOException {
        BufferedReader br=new BufferedReader(new FileReader(new File(dataFile)));
        String line="";
        br.readLine();
        while((line=br.readLine())!=null){

            String[] inputData=line.split("\t");
            this.gold.add(inputData[1]);
            this.sentences.add(inputData[0]);
        }
    }

    public void classify(String dataFile,String modelDir,Map<String, String> categoryMap) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        readData(dataFile);
        BertRelation classifier = new BertRelation();
        long startTime=System.currentTimeMillis();
        Criteria<RCInput, String> criteria = Criteria.builder()
                .setTypes(RCInput.class, String.class)
//                .optModelPath(Paths.get("build/pytorch/bertqa/")) // search in local folder
//                .optModelPath(Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\biobert_ts\\"))
//                .optModelPath(Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\pubmedbert_ts\\"))
                .optModelPath(Paths.get(modelDir))
//                .optModelPath(Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\bertqa\\"))
//                .optModelPath(Paths.get("/Users/qwei1/Workspaces/clampbertjava/bertqa/"))
                .optTranslator(classifier)
                .optProgress(new ProgressBar()).build();

        ZooModel model = criteria.loadModel();

        for(int i=0;i<this.sentences.size();i++){
            String predictResult = null;


            RCInput input = new RCInput(sentences.get(i));
//            input=new RCInput("@ chem $ inhibits proteinase and selected @ gene $ activities of the catalytic core proteasome at low micromolar concentrations .");

            try (Predictor<RCInput, String> predictor = model.newPredictor(classifier)) {
                predictResult = predictor.predict(input);
            }

            prediction.add(categoryMap.get(predictResult));

            if(i%100==0){
                System.out.println(i+1+" "+(System.currentTimeMillis()-startTime)/1000.0);
            }
        }

        calculatePerformance(gold,prediction,"NO");
//        System.out.println(question);
//        System.out.println(predictResult);
    }

    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {

        String categories="ACTIVATOR\nAGONIST\nAGONIST-ACTIVATOR\nAGONIST-INHIBITOR\nANTAGONIST\nDIRECT-REGULATOR\nINDIRECT-DOWNREGULATOR\nINDIRECT-UPREGULATOR\nINHIBITOR\nNO\nPART-OF\nPRODUCT-OF\nSUBSTRATE\nSUBSTRATE_PRODUCT-OF";
        String[] categoryArray=categories.split("\n");
        Map<String,String> categoryMap=IntStream.range(0,categoryArray.length).boxed().collect(Collectors.toMap(i->String.valueOf(i),i->categoryArray[i]));

        String dataFile="D:\\Shared\\pubmedbert_re\\dev.tsv";
        String modelDir="C:\\Users\\cherub\\Downloads\\test_djl\\pubmedbert_ts";
        TestBertRelationClassification classifier=new TestBertRelationClassification();
//        System.out.println(System.mapLibraryName("torch"));
        classifier.classify(dataFile,modelDir,categoryMap);
    }
}
