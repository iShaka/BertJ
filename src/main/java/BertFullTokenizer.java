
import ai.djl.modality.nlp.bert.*;

import java.util.*;
import java.util.stream.*;

public class BertFullTokenizer extends BertTokenizer{

    //bert字典
    private Map<String, Integer> vocab;
    //bert基础的分词类
    private BertBasicTokenizer basicTokenizer;
    //bert的wordpiece分词类(对中文来说没啥意义，中文之前的分词已经是最小了)
    private BertWordpieceTokenizer wordpieceTokenizer;

    public BertFullTokenizer(Map<String, Integer> vocab, boolean doLowerCase) {
        this.vocab = vocab;
        this.basicTokenizer = new BertBasicTokenizer(doLowerCase);
        this.wordpieceTokenizer = new BertWordpieceTokenizer(vocab);
    }

    //分词,首先基础的分词（中文就是汉字两边加空格），再用词典进行wordpiece分词
    public List<String> tokenize(String text) {
        List<String> splitTokens = new ArrayList<String>();

        for(String token : basicTokenizer.tokenize(text)) {
            for(String subToken : wordpieceTokenizer.tokenize(token)) {
                splitTokens.add(subToken);
            }
        }

        return splitTokens;
    }
    //把字符转化成id
    public List<Integer>  convertTokensToIds(List<String> tokens) {
        List<Integer> outputIds = new ArrayList<Integer>();
        for(String token : tokens) {
            Integer tokenId=this.vocab.get(token);
            if (tokenId==null){
                tokenId=0;//unk
                System.out.println("unknowToken"+token);
            }
            outputIds.add(tokenId);

        }
        return outputIds;
    }

    public BertToken encode(String text) {
        List<String> tToken = tokenize(text);
        int validLength = tToken.size();
        tToken.add(0, "[CLS]");
        tToken.add("[SEP]");
        List<String> tokens = new ArrayList<>(tToken);

        int tokenTypeStartIdx = tToken.size();
        long[] tokenTypeArr = new long[tokens.size()];
        Arrays.fill(tokenTypeArr, tokenTypeStartIdx, tokenTypeArr.length, 1);

        long[] attentionMaskArr = new long[tokens.size()];
        Arrays.fill(attentionMaskArr, 1);

        return new BertToken(
                tokens,
                Arrays.stream(tokenTypeArr).boxed().collect(Collectors.toList()),
                Arrays.stream(attentionMaskArr).boxed().collect(Collectors.toList()),
                validLength);
    }


    public BertToken rawEncode(String text) {
        List<String> tToken = tokenize(text);
        int validLength = tToken.size();

        List<String> tokens = new ArrayList<>(tToken);

        int tokenTypeStartIdx = tToken.size();
        long[] tokenTypeArr = new long[tokens.size()];
        Arrays.fill(tokenTypeArr, tokenTypeStartIdx, tokenTypeArr.length, 1);

        long[] attentionMaskArr = new long[tokens.size()];
        Arrays.fill(attentionMaskArr, 1);

        return new BertToken(
                tokens,
                Arrays.stream(tokenTypeArr).boxed().collect(Collectors.toList()),
                Arrays.stream(attentionMaskArr).boxed().collect(Collectors.toList()),
                validLength);
    }

    /**
     * Encodes questions and paragraph sentences with max length.
     *
     * @param text the input text
     * @param maxLength the maxLength
     * @return BertToken
     */
    public BertToken encode(String text, int maxLength) {
        BertToken bertToken = encode(text);
        return new BertToken(
                pad(bertToken.getTokens(), "[PAD]", maxLength),
                pad(bertToken.getTokenTypes(), 0L, maxLength),
                pad(bertToken.getAttentionMask(), 0L, maxLength),
                bertToken.getValidLength());
    }

}
