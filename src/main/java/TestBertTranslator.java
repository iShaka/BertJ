import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.nlp.qa.*;
import ai.djl.repository.zoo.*;
import ai.djl.training.util.*;
import ai.djl.translate.*;

import java.io.*;
import java.nio.file.*;

public class TestBertTranslator {

    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
//        DownloadUtils.download("https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/pytorch/bertqa/0.0.1/trace_bertqa.pt.gz", "build/pytorch/bertqa/bertqa.pt"
//                , new ProgressBar());

        BertTranslator translator = new BertTranslator();

        Criteria<QAInput, String> criteria = Criteria.builder()
                .setTypes(QAInput.class, String.class)
//                .optModelPath(Paths.get("build/pytorch/bertqa/")) // search in local folder
//                .optModelPath(Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\biobert_ts\\"))
                .optModelPath(Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\pubmedbert_ts\\"))
//                .optModelPath(Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\bertqa\\"))
//                .optModelPath(Paths.get("/Users/qwei1/Workspaces/clampbertjava/bertqa/"))
                .optTranslator(translator)
                .optProgress(new ProgressBar()).build();

        ZooModel model = criteria.loadModel();

        String predictResult = null;

        String question = "When did BBC Japan start broadcasting?";
        String resourceDocument = "BBC Japan was a general entertainment Channel.\n" +
                "Which operated between December 2004 and April 2006.\n" +
                "It ceased operations after its Japanese distributor folded.";

        question="@ chem $ inhibits proteinase and selected @ gene $ activities of the catalytic core proteasome at low micromolar concentrations .";
//        question="@chem$ inhibits proteinase and selected @gene$ activities of the catalytic core proteasome at low micromolar concentrations .";
        resourceDocument="";

        QAInput input = new QAInput(question, resourceDocument);

// Create a Predictor and use it to predict the output
        try (Predictor<QAInput, String> predictor = model.newPredictor(translator)) {
            predictResult = predictor.predict(input);
        }

//        System.out.println(question);
        System.out.println(predictResult);
    }
}
