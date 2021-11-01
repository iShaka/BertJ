import com.google.common.collect.ImmutableSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class TokenUtils {
    private static Set<Integer> control_char_sets = ImmutableSet.of(
            (int)Character.CONTROL,
            (int)Character.FORMAT,
            (int)Character.PRIVATE_USE,
            (int)Character.SURROGATE,
            (int)Character.UNASSIGNED);

    private static Set<Integer> punctuation_char_sets = ImmutableSet.of(
            (int)Character.CONNECTOR_PUNCTUATION,
            (int)Character.DASH_PUNCTUATION,
            (int)Character.END_PUNCTUATION,
            (int)Character.FINAL_QUOTE_PUNCTUATION,
            (int)Character.INITIAL_QUOTE_PUNCTUATION,
            (int)Character.OTHER_PUNCTUATION,
            (int)Character.START_PUNCTUATION);


    //判断是否是中文字符
    public static boolean isChineseChar(char c) {
        if ((c >= 0x4E00 && c <= 0x9FFF) ||
                (c >= 0x3400 && c <= 0x4DBF) ||
                (c >= 0x20000 && c <= 0x2A6DF) ||
                (c >= 0x2A700 && c <= 0x2B73F) ||
                (c >= 0x2B740 && c <= 0x2B81F) ||
                (c >= 0x2B820 && c <= 0x2CEAF) ||
                (c >= 0xF900 && c <= 0xFAFF) ||
                (c >= 0x2F800 && c <= 0x2FA1F)) {
            return true;
        }
        return false;
    }

    //文本预处理，去除一些无意义的字符以及whitespace
    public static String cleanText(String text) {
        StringBuffer outStringBuf = new StringBuffer("");

        for(int i=0; i<text.length(); i++) {
            char c = text.charAt(i);
            if(c == 0 || c == 0xfffd || isControl(c)) {
                continue;
            }

            if(isWhiteSpace(c)) {
                outStringBuf.append(" ");
            }
            else {
                outStringBuf.append(c);
            }
        }
        return outStringBuf.toString();
    }

    //检查字符char是否是控制字符
    public static boolean isControl(char c) {
        // count them as whitespace
        if(c == '\t' || c == '\n' || c == '\r') {
            return false;
        }
        // Unicode specification starts with "C"
        if (control_char_sets.contains(Character.getType(c))) {
            return true;
        }

        return false;
    }

    //检查字符是否是空白符
    public static boolean isWhiteSpace(char c) {
        if(c == ' ' || c == '\t' || c == '\n' || c == '\r'){
            return true;
        }

        return false;
    }

    //标点符号split
    public static List<String> runSplitOnPunc(String token){
        //存放的是标点符号
        List<List<Character>> result = new ArrayList<List<Character>>();
        int length = token.length();
        int i =0;
        boolean startNewWord = true;
        while(i < length) {
            char c = token.charAt(i);
            if(isPunctuation(c)) {
                List<Character> list = Arrays.asList(c);
                result.add(list);
                startNewWord = true;
            }
            else {
                if(startNewWord) {
                    result.add(new ArrayList<Character>());
                }
                startNewWord = false;
                result.get(result.size()-1).add(c);
            }
            i += 1;
        }

        List<String> res = new ArrayList<String>();
        for(int j=0; j<result.size(); j++) {
            String str = "";
            for(int k=0; k<result.get(j).size(); k++) {
                str += result.get(j).get(k);
            }
            res.add(str);
        }
        return res;
    }

    //判断是否是标点符号
    public static boolean isPunctuation(char c) {
        if((c >= 33 && c <= 47) || (c >=58 && c <= 64) ||
                (c >= 91 && c <= 96) || (c >= 123 && c<= 126)){
            return true;
        }
        // Unicode specification starts with "P" in bert-tensorflow
        if (punctuation_char_sets.contains(Character.getType(c))) {
            return true;
        }
        return false;
    }

    //按照空白符进行分词
    public static List<String> whiteSpaceTokenize(String text) {
        List<String> result = new ArrayList<String>();

        text = text.trim();
        if(null == text) {
            return result;
        }
        String[] tokens = text.split(" ");
        result = Arrays.asList(tokens);

        return result;
    }

    //对中文进行分词,是中文就两边加空格
    public static String tokenizeChineseChars(String cleanText) {
        StringBuffer outStrBuf = new StringBuffer();

        for(int i=0; i< cleanText.length(); i++) {
            char c = cleanText.charAt(i);
            if(isChineseChar(c)) {
                outStrBuf.append(" ");
                outStrBuf.append(c);
                outStrBuf.append(" ");
            }else{
                outStrBuf.append(c);
            }
        }

        return outStrBuf.toString();
    }

}
