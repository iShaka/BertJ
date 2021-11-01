import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.repository.zoo.*;
import ai.djl.training.util.*;
import ai.djl.translate.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

public class TestBertNER {

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

    List<NERExample> examples =new ArrayList<>();

    public void readData(String dataFile) throws IOException {
        BufferedReader br=new BufferedReader(new FileReader(new File(dataFile)));
        String line="";
        br.readLine();
        NERExample example=new NERExample();
        while((line=br.readLine())!=null){
            if(!line.trim().equals("")){
                String[] t=line.trim().split("\t");
                example.getTokens().add(t[0]);
                example.getLabels().add(t[1]);
                continue;
            }
            examples.add(example);
            example=new NERExample();

        }
        br.close();
    }

    public void classify(String dataFile,String modelDir,Map<String, String> categoryMap) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        readData(dataFile);
        System.out.println("total instances "+this.examples.size());
        BertNER classifier = new BertNER();
        long startTime=System.currentTimeMillis();
        Criteria<NERExample, NERExample> criteria = Criteria.builder()
                .setTypes(NERExample.class, NERExample.class)
//                .optModelPath(Paths.get("build/pytorch/bertqa/")) // search in local folder
//                .optModelPath(Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\biobert_ts\\"))
//                .optModelPath(Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\pubmedbert_ts\\"))
                .optModelPath(Paths.get(modelDir))
//                .optModelPath(Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\bertqa\\"))
//                .optModelPath(Paths.get("/Users/qwei1/Workspaces/clampbertjava/bertqa/"))
                .optTranslator(classifier)
                .optProgress(new ProgressBar()).build();

        ZooModel model = criteria.loadModel();


//        String text="School of Nursing - Main Building Houston , TX LUMMUS , TENA M January 11 , 1946 Date of Birth Female Sex 113203 Patient Id 1209 MODENA DR PEARLAND , TX 77581 AddressEnglish ( preferred ) Language White Race Not Hispanic or Latino EthnicitySummary of Care Clinical Content Allergies and Adverse Reactions";
//        NEROutput predictResult = null;
//        NERInput input = new NERInput(text);

//        try (Predictor<NERInput, NEROutput> predictor = model.newPredictor(classifier)) {
//                predictResult = predictor.predict(input);
//            }
//        System.out.println(predictResult.getOutput());

        for(int i=0;i<this.examples.size();i++){
            NERExample predictResult = null;


            NERExample input = examples.get(i);
//            input=new RCInput("@ chem $ inhibits proteinase and selected @ gene $ activities of the catalytic core proteasome at low micromolar concentrations .");

            try (Predictor<NERExample, NERExample> predictor = model.newPredictor(classifier)) {
                predictResult = predictor.predict(input);
            }


            for(String bioIdx:input.getWpLabelIdxs()){
                input.getWpLabels().add(categoryMap.get(bioIdx));
            }

            input.generatePredLabels();

            if(i%100==0){
                System.out.println(i+" "+(System.currentTimeMillis()-startTime)/1000.0);
            }
//            if(i==1000){break;}
        }
//
//        calculatePerformance(gold,prediction,"NO");
//        System.out.println(question);
//        System.out.println(predictResult);
    }

    public void writePrediction(String outFile) throws IOException {
        BufferedWriter bw=new BufferedWriter(new FileWriter(new File(outFile)));
        int n=0;
        for(NERExample example:examples){

            for(int i=0;i<example.getTokens().size();i++){
                bw.write(example.getTokens().get(i)+"\t"+example.getPredLabels().get(i)+"\t"+example.getLabels().get(i)+"\n");
            }
            bw.write("\n");

//            if(n==1000){break;}
            n+=1;
        }
        bw.close();
    }

    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {

        String categories="O,B-ZIP,I-ZIP,B-PHONE,I-PHONE,B-PATIENT,I-PATIENT,B-FAX,I-FAX,B-STREET,I-STREET,B-MEDICALRECORD,I-MEDICALRECORD,B-CITY,I-CITY,B-USERNAME,I-USERNAME,B-AGE,I-AGE,B-LOCATION-OTHER,I-LOCATION-OTHER,B-HEALTHPLAN,I-HEALTHPLAN,B-EMAIL,I-EMAIL,B-IDNUM,I-IDNUM,B-BIOID,I-BIOID,B-DEVICE,I-DEVICE,B-DOCTOR,I-DOCTOR,B-URL,I-URL,B-HOSPITAL,I-HOSPITAL,B-STATE,I-STATE,B-ORGANIZATION,I-ORGANIZATION,B-COUNTRY,I-COUNTRY,B-DATE,I-DATE,B-PROFESSION,I-PROFESSION";
        String[] categoryArray=categories.split(",");
        Map<String,String> categoryMap=IntStream.range(0,categoryArray.length).boxed().collect(Collectors.toMap(i->String.valueOf(i),i->categoryArray[i]));

        String dataFile="D:\\Shared\\bert_ner\\test.txt";
        String modelDir="C:\\Users\\cherub\\Downloads\\test_djl\\bert_ner_ts";
        String outFile="C:\\Users\\cherub\\Downloads\\test_djl\\bert_ner_ts\\predictions.txt";
        TestBertNER classifier=new TestBertNER();
//        System.out.println(System.mapLibraryName("torch"));
        classifier.classify(dataFile,modelDir,categoryMap);
        classifier.writePrediction(outFile);
    }
}
