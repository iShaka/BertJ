
import ai.djl.modality.nlp.bert.*;
import ai.djl.modality.nlp.preprocess.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.*;

public class BertWordpieceTokenizer extends BertTokenizer {

    //bert的字典
    private Map<String, Integer> vocab;
    //未登录词
    private String unkToken = "[UNK]";
    //最大长度
    private int maxInputCharsPerWord = 200;

    public BertWordpieceTokenizer(Map<String, Integer> vocab) {
        this.vocab = vocab;
    }


    //把一段文字切分成word piece。这其实是贪心的最大正向匹配算法
    @Override
    public List<String> tokenize(String text) {
        //空白符分词
        List<String> tokens = whiteSpaceTokenize(text);

        List<String> outputTokens = new ArrayList<String>();
        for(String token : tokens){
            int length = token.length();
            //最大长度分词
            if(length > this.maxInputCharsPerWord) {
                outputTokens.add(this.unkToken);
                continue;
            }

            boolean isBad = false;
            int start = 0;
            List<String> subTokens = new ArrayList<String>();

            while(start < length) {
                int end = length;
                String curSubStr = null;
                while(start < end) {
                    String subStr = token.substring(start, end);
                    if(start > 0) {
                        subStr = "##" + subStr;
                    }
                    if(this.vocab.containsKey(subStr)) {
                        curSubStr = subStr;
                        break;
                    }
                    end -= 1;
                }
                if(null == curSubStr) {
                    isBad = true;
                    break;
                }
                subTokens.add(curSubStr);
                start = end;
            }

            if(isBad) {
                outputTokens.add(this.unkToken);
            }
            else {
                outputTokens.addAll(subTokens);
            }

        }
        return outputTokens;
    }
    //空白符分词
    private List<String> whiteSpaceTokenize(String text) {
        List<String> result = new ArrayList<String>();

        text = text.trim();
        if(null == text){
            return result;
        }
        String[] tokens = text.split(" ");
        result = Arrays.asList(tokens);

        return result;
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
