import ai.djl.*;
import ai.djl.modality.nlp.bert.*;
import ai.djl.ndarray.*;
import ai.djl.translate.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class BertRelation implements Translator<RCInput, String> {
    private List<String> tokens;
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
//        Path path=Paths.get(model.getModelPath().getParent().toString(),"vocab.txt");
        Path path = Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\pubmedbert_ts\\vocab.txt");
        vocabulary = BertVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(path)
                .optUnknownToken("[UNK]")
                .build();
        Map<String,Integer> tokenMap=vocabulary.getTokenMap();
        tokenizer = new BertFullTokenizer(tokenMap,true);
    }

    @Override
    public NDList processInput(TranslatorContext ctx, RCInput input) {
        BertToken token =
                tokenizer.encode(
                        input.getText().toLowerCase());
        // get the encoded tokens that would be used in precessOutput
        tokens = token.getTokens();

        NDManager manager = ctx.getNDManager();
        // map the tokens(String) to indices(long)
//        System.out.println(tokens);
        long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
        long[] tokenType = token.getTokenTypes().stream().mapToLong(i -> i).toArray();

        NDArray indicesArray = manager.create(indices);
        NDArray tokenTypeArray = manager.create(tokenType);
        return new NDList(indicesArray, tokenTypeArray);

//        long[] indices2=new long[indices.length-1];
//        long[] tokenType2=new long[indices.length-1];
//        for(int i=0;i<indices2.length;i++){
//            indices2[i]=indices[i];
//            tokenType2[i]=tokenType[i];
//        }
//        for (Long l : indices2){
//            System.out.print(l +" , ");}
//        NDArray indicesArray = manager.create(indices2);
//        NDArray tokenTypeArray = manager.create(tokenType2);
//        return new NDList(indicesArray, tokenTypeArray);
    }


    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        NDArray startLogits = list.get(0);
//        System.out.println(startLogits);
        int startIdx = (int) startLogits.argMax().getLong();
        return String.valueOf(startIdx);
    }

}