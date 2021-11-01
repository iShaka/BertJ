import java.util.*;

public class NERExample {

    private String text="";

    private List<String> tokens=new ArrayList<>();
    private List<String> wordPieces=new ArrayList<>();
    private List<Integer[]> tokenToWP=new ArrayList<>();
    private Map<Integer,Integer> wpToToken=new HashMap<>();
    private List<String> wpLabels=new ArrayList<>();

    private List<String> wpLabelIdxs=new ArrayList<>();

    private List<String> labels=new ArrayList<>();



    List<String> predLabels =new ArrayList<>();

    public NERExample() {

    }

    public void setBertMarker(){
        tokens.add(0,"[CLS]");
        tokens.add("[SEP]");
    }

    public void updateWpToToken(){
        for(int i=0;i<tokenToWP.size();i++){
            Integer[] range=tokenToWP.get(i);
            wpToToken.put(range[0],i);
            wpToToken.put(range[1],i);
        }
    }

    public void generatePredLabels(){
        for(int i=0;i<tokens.size();i++){
            Integer[] range=tokenToWP.get(i);
            String label=wpLabels.get(range[0]);
            predLabels.add(label);
        }
    }

    public List<String> getPredLabels() {
        return predLabels;
    }

    public void setPredLabels(List<String> predLabels) {
        this.predLabels = predLabels;
    }


    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public List<String> getTokens() {
        return tokens;
    }

    public void setTokens(List<String> tokens) {
        this.tokens = tokens;
    }

    public List<String> getWordPieces() {
        return wordPieces;
    }

    public void setWordPieces(List<String> wordPieces) {
        this.wordPieces = wordPieces;
    }

    public List<Integer[]> getTokenToWP() {
        return tokenToWP;
    }

    public void setTokenToWP(List<Integer[]> tokenToWP) {
        this.tokenToWP = tokenToWP;
    }

    public Map<Integer, Integer> getWpToToken() {
        return wpToToken;
    }

    public void setWpToToken(Map<Integer, Integer> wpToToken) {
        this.wpToToken = wpToToken;
    }

    public List<String> getLabels() {
        return labels;
    }

    public void setLabels(List<String> labels) {
        this.labels = labels;
    }

    public List<String> getWpLabels() {
        return wpLabels;
    }

    public void setWpLabels(List<String> wpLabels) {
        this.wpLabels = wpLabels;
    }

    public List<String> getWpLabelIdxs() {
        return wpLabelIdxs;
    }

    public void setWpLabelIdxs(List<String> wpLabelIdxs) {
        this.wpLabelIdxs = wpLabelIdxs;
    }


}
