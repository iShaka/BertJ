import ai.djl.*;
import ai.djl.modality.nlp.*;
import ai.djl.modality.nlp.bert.*;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.ndarray.*;
import ai.djl.translate.*;

import java.io.*;
import java.nio.file.*;
import java.util.List;

public class BertTranslator implements Translator<QAInput, String> {
    private List<String> tokens;
    private Vocabulary vocabulary;
    private BertTokenizer tokenizer;

    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
//        Path path = Paths.get("build/pytorch/bertqa/vocab.txt");
//        Path path = Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\biobert\\vocab.txt");
        Path path = Paths.get("C:\\Users\\cherub\\Downloads\\test_djl\\pubmedbert_ts\\vocab.txt");
        vocabulary = SimpleVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(path)
                .optUnknownToken("[UNK]")
                .build();
        tokenizer = new BertTokenizer();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) {
        BertToken token =
                tokenizer.encode(
                        input.getQuestion().toLowerCase(),
                        input.getParagraph().toLowerCase());
        // get the encoded tokens that would be used in precessOutput
        tokens = token.getTokens();

        NDManager manager = ctx.getNDManager();
        // map the tokens(String) to indices(long)
        System.out.println(tokens);
        long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();

        long[] attentionMask = token.getAttentionMask().stream().mapToLong(i -> i).toArray();
        long[] tokenType = token.getTokenTypes().stream().mapToLong(i -> i).toArray();

        long[] indices2=new long[indices.length-1];
        long[] tokenType2=new long[indices.length-1];
        for(int i=0;i<indices2.length;i++){
            indices2[i]=indices[i];
            tokenType2[i]=tokenType[i];
        }
        for (Long l : indices2){
            System.out.print(l +" , ");}
        NDArray indicesArray = manager.create(indices2);
//        NDArray attentionMaskArray =
//                manager.create(attentionMask);
        NDArray tokenTypeArray = manager.create(tokenType2);
        // The order matters
//        return new NDList(indicesArray, attentionMaskArray, tokenTypeArray);
        return new NDList(indicesArray, tokenTypeArray);
    }

//    @Override
//    public NDList processInput(TranslatorContext ctx, QAInput input) {
//        BertToken token =
//                tokenizer.encode(
//                        input.getQuestion().toLowerCase(),
//                        input.getParagraph().toLowerCase());
//        // get the encoded tokens that would be used in precessOutput
//        tokens = token.getTokens();
//
//        NDManager manager = ctx.getNDManager();
//        // map the tokens(String) to indices(long)
//        System.out.println(tokens);
//        long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
//        for (Long l : indices){
//            System.out.print(l +" , ");}
//        long[] attentionMask = token.getAttentionMask().stream().mapToLong(i -> i).toArray();
//        long[] tokenType = token.getTokenTypes().stream().mapToLong(i -> i).toArray();
//
//
//
//        NDArray indicesArray = manager.create(indices);
//        NDArray attentionMaskArray =
//                manager.create(attentionMask);
//        NDArray tokenTypeArray = manager.create(tokenType);
//        // The order matters
////        return new NDList(indicesArray, attentionMaskArray, tokenTypeArray);
//        return new NDList(indicesArray, tokenTypeArray);
//    }

//    @Override
//    public String processOutput(TranslatorContext ctx, NDList list) {
//        NDArray startLogits = list.get(0);
//        NDArray endLogits = list.get(1);
//        int startIdx = (int) startLogits.argMax().getLong();
//        int endIdx = (int) endLogits.argMax().getLong();
//        return tokens.subList(startIdx, endIdx + 1).toString();
//    }

    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        NDArray startLogits = list.get(0);
        System.out.println(startLogits);
//        NDArray endLogits = list.get(1);
        int startIdx = (int) startLogits.argMax().getLong();
//        int endIdx = (int) endLogits.argMax().getLong();
        return String.valueOf(startIdx);
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }
}