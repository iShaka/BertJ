import ai.djl.modality.nlp.*;
import ai.djl.modality.nlp.bert.*;

import java.io.*;
import java.nio.file.*;

public class TestBioBert {
    public static void main(String[] args) throws IOException {
        String sent="I have an apple.";
        String sent2="";
        BertTokenizer tokenizer = new BertTokenizer();
        BertToken token = tokenizer.encode(sent.toLowerCase(), sent2.toLowerCase());
//        Path path = Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\biobert\\vocab.txt");
        Path path = Paths.get("build/pytorch/bertqa/vocab.txt");
        SimpleVocabulary vocabulary = SimpleVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(path)
                .optUnknownToken("[UNK]")
                .build();
    }
}
