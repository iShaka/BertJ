import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

import ai.djl.*;
import ai.djl.modality.nlp.preprocess.Tokenizer;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.*;
import ai.djl.inference.*;
import ai.djl.translate.*;
import ai.djl.training.util.*;
import ai.djl.repository.zoo.*;
import ai.djl.modality.nlp.*;
import ai.djl.modality.nlp.qa.*;
import ai.djl.modality.nlp.bert.*;

public class TestPytorchBert {

    public static void main(String[] args) throws IOException {
        String question = "When did BBC Japan start broadcasting?";
        String resourceDocument = "BBC Japan was a general entertainment Channel.\n" +
                "Which operated between December 2004 and April 2006.\n" +
                "It ceased operations after its Japanese distributor folded.";

        QAInput input = new QAInput(question, resourceDocument);

        BertTokenizer tokenizer = new BertTokenizer();
        List<String> tokenQ = tokenizer.tokenize(question.toLowerCase());
        List<String> tokenA = tokenizer.tokenize(resourceDocument.toLowerCase());

        System.out.println("Question Token: " + tokenQ);
        System.out.println("Answer Token: " + tokenA);

        BertToken token = tokenizer.encode(question.toLowerCase(), resourceDocument.toLowerCase());
        System.out.println("Encoded tokens: " + token.getTokens());
        System.out.println("Encoded token type: " + token.getTokenTypes());
        System.out.println("Valid length: " + token.getValidLength());

//        DownloadUtils.download("https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/pytorch/bertqa/0.0.1/bert-base-uncased-vocab.txt.gz", "build/pytorch/bertqa/vocab.txt", new ProgressBar());


        Path path = Paths.get("build/pytorch/bertqa/vocab.txt");
        SimpleVocabulary vocabulary = SimpleVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(path)
                .optUnknownToken("[UNK]")
                .build();

        long index = vocabulary.getIndex("car");
        String tokenString = vocabulary.getToken(2482);
        System.out.println("The index of the car is " + index);
        System.out.println("The token of the index 2482 is " + tokenString);
    }
}
