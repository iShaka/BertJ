
import org.apache.commons.lang3.StringUtils;

import java.util.List;


public class BertBasicTokenizer {
    private boolean doLowerCase;


    public BertBasicTokenizer(boolean doLowerCase) {
        this.doLowerCase = doLowerCase;
    }

    //分词
    public List<String> tokenize(String text){
        String cleanText = TokenUtils.cleanText(text);
        String chineseTokens = TokenUtils.tokenizeChineseChars(cleanText);
        List<String> origTokens = TokenUtils.whiteSpaceTokenize(chineseTokens);

        String str = "";
        for(String token : origTokens) {
            if (doLowerCase) {
                token = token.toLowerCase();
                token = StringUtils.stripAccents(token); //去除无意义的词语
            }
            //标点符号替换成空白符
            List<String> list = TokenUtils.runSplitOnPunc(token);
            for(int i=0; i<list.size(); i++) {
                str += list.get(i) + " ";
            }
        }
        List<String> resTokens = TokenUtils.whiteSpaceTokenize(str);
        return resTokens;
    }


    public static void main(String[] args) {

        System.out.print("hello world.");
    }


}
