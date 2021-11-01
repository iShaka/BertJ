import ai.djl.*;
import ai.djl.modality.nlp.bert.*;
import ai.djl.ndarray.*;
import ai.djl.translate.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

public class BertNER implements Translator<NERExample, NERExample> {
    private NERExample example;
    private BertVocabulary vocabulary;
    private BertFullTokenizer tokenizer;

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }

    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
//        Path path = Paths.get("build/pytorch/bertqa/vocab.txt");
//        Path path = Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\biobert\\vocab.txt");
//        Path path = Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\pubmedbert_ts\\vocab.txt");
        Path path=Paths.get(model.getModelPath().getParent().toString(),"vocab.txt");

//        Path path = Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\bert_ner_ts\\vocab.txt");
        vocabulary = BertVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(path)
                .optUnknownToken("[UNK]")
                .build();
        Map<String,Integer> tokenMap=vocabulary.getTokenMap();
        tokenizer = new BertFullTokenizer(tokenMap,false);
    }

    @Override
    public NDList processInput(TranslatorContext ctx, NERExample input) {

        this.example=input;
//        example.setBertMarker();
        for(int i=0;i<example.getTokens().size();i++){
            BertToken token = tokenizer.rawEncode(example.getTokens().get(i));
            List<String> wordPieces=token.getTokens();
            Integer[] range=new Integer[2];
            range[0]=example.getWordPieces().size();
            range[1]=range[0]+wordPieces.size();
            example.getWordPieces().addAll(wordPieces);
            example.getTokenToWP().add(range);
        }
        example.updateWpToToken();

        List<String> wpWithMarker=example.getWordPieces().stream().collect(Collectors.toList());
        wpWithMarker.add("[SEP]");
        wpWithMarker.add(0,"[CLS]");

        NDManager manager = ctx.getNDManager();

        long[] indices = wpWithMarker.stream().mapToLong(vocabulary::getIndex).toArray();
        long[] tokenType = IntStream.range(0,wpWithMarker.size()).mapToLong(i->0).toArray();

        NDArray indicesArray = manager.create(indices);
        NDArray tokenTypeArray = manager.create(tokenType);
        return new NDList(indicesArray, tokenTypeArray);

    }


    @Override
    public NERExample processOutput(TranslatorContext ctx, NDList list) {

        NDArray logits = list.get(0);
        long len=logits.getShape().get(0);
        for(int i=1;i<len-1;i++){
            example.getWpLabelIdxs().add(String.valueOf(logits.argMax(1).get(i).getLong()));
        }
        return example;
    }

}