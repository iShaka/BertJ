import ai.djl.modality.nlp.*;
import ai.djl.util.*;

import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;

public class BertVocabulary  implements Vocabulary {
    private Map<String, BertVocabulary.TokenInfo> tokens = new ConcurrentHashMap<>();
    private List<String> indexToToken = new ArrayList<>();
    private Set<String> reservedTokens;
    private int minFrequency;
    private String unknownToken;

    /**
     * Create a {@code BertVocabulary} object with a {@link BertVocabulary.Builder}.
     *
     * @param builder the {@link BertVocabulary.Builder} to build the vocabulary with
     */
    public BertVocabulary(BertVocabulary.Builder builder) {
        reservedTokens = builder.reservedTokens;
        minFrequency = builder.minFrequency;
        unknownToken = builder.unknownToken;
        reservedTokens.add(unknownToken);
        addTokens(reservedTokens);
        for (List<String> sentence : builder.sentences) {
            for (String word : sentence) {
                addWord(word);
            }
        }
    }

    /**
     * Create a {@code BertVocabulary} object with the given list of tokens.
     *
     * @param tokens the {@link List} of tokens to build the vocabulary with
     */
    public BertVocabulary(List<String> tokens) {
        reservedTokens = new HashSet<>();
        minFrequency = 10;
        unknownToken = "<unk>";
        reservedTokens.add(unknownToken);
        addTokens(reservedTokens);
        addTokens(tokens);
    }

    private void addWord(String token) {
        if (reservedTokens.contains(token)) {
            return;
        }
        BertVocabulary.TokenInfo tokenInfo = tokens.getOrDefault(token, new BertVocabulary.TokenInfo());
        if (++tokenInfo.frequency == minFrequency) {
            tokenInfo.index = tokens.size();
            indexToToken.add(token);
        }
        tokens.put(token, tokenInfo);
    }

    private void addTokens(Collection<String> tokens) {
        for (String token : tokens) {
            BertVocabulary.TokenInfo tokenInfo = new BertVocabulary.TokenInfo();
            tokenInfo.frequency = Integer.MAX_VALUE;
            tokenInfo.index = indexToToken.size();
            indexToToken.add(token);
            this.tokens.put(token, tokenInfo);
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean contains(String token) {
        return tokens.containsKey(token);
    }

    /** {@inheritDoc} */
    @Override
    public String getToken(long index) {
        if (index < 0 || index >= indexToToken.size()) {
            return unknownToken;
        }
        return indexToToken.get((int) index);
    }

    /** {@inheritDoc} */
    @Override
    public long getIndex(String token) {
        if (tokens.containsKey(token)) {
            BertVocabulary.TokenInfo tokenInfo = tokens.get(token);
            if (tokenInfo.frequency >= minFrequency) {
                return tokenInfo.index;
            }
        }
        return tokens.get(unknownToken).index;
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        return tokens.size();
    }

    /**
     * Creates a new builder to build a {@code BertVocabulary}.
     *
     * @return a new builder
     */
    public static BertVocabulary.Builder builder() {
        return new BertVocabulary.Builder();
    }

    /** Builder class that is used to build the {@link BertVocabulary}. */
    public static final class Builder {

        List<List<String>> sentences = new ArrayList<>();
        Set<String> reservedTokens = new HashSet<>();
        int minFrequency = 10;
        String unknownToken = "<unk>";

        private Builder() {}

        /**
         * Sets the optional parameter that specifies the minimum frequency to consider a token to
         * be part of the {@link BertVocabulary}. Defaults to 10.
         *
         * @param minFrequency the minimum frequency to consider a token to be part of the {@link
         *     BertVocabulary}
         * @return this {@code VocabularyBuilder}
         */
        public BertVocabulary.Builder optMinFrequency(int minFrequency) {
            this.minFrequency = minFrequency;
            return this;
        }

        /**
         * Sets the optional parameter that specifies the unknown token's string value.
         *
         * @param unknownToken the string value of the unknown token
         * @return this {@code VocabularyBuilder}
         */
        public BertVocabulary.Builder optUnknownToken(String unknownToken) {
            this.unknownToken = unknownToken;
            return this;
        }

        /**
         * Sets the optional parameter that sets the list of reserved tokens.
         *
         * @param reservedTokens the list of reserved tokens
         * @return this {@code VocabularyBuilder}
         */
        public BertVocabulary.Builder optReservedTokens(Collection<String> reservedTokens) {
            this.reservedTokens.addAll(reservedTokens);
            return this;
        }

        /**
         * Adds the given sentence to the {@link BertVocabulary}.
         *
         * @param sentence the sentence to be added
         * @return this {@code VocabularyBuilder}
         */
        public BertVocabulary.Builder add(List<String> sentence) {
            this.sentences.add(sentence);
            return this;
        }

        /**
         * Adds the given list of sentences to the {@link BertVocabulary}.
         *
         * @param sentences the list of sentences to be added
         * @return this {@code VocabularyBuilder}
         */
        public BertVocabulary.Builder addAll(List<List<String>> sentences) {
            this.sentences.addAll(sentences);
            return this;
        }

        /**
         * Adds a text vocabulary to the {@link BertVocabulary}.
         *
         * <pre>
         *   Example text file(vocab.txt):
         *   token1
         *   token2
         *   token3
         *   will be mapped to index of 0 1 2
         * </pre>
         *
         * @param path the path to the text file
         * @return this {@code VocabularyBuilder}
         * @throws IOException if failed to read vocabulary file
         */
        public BertVocabulary.Builder addFromTextFile(Path path) throws IOException {
            add(Utils.readLines(path, true));
            return this;
        }

        /**
         * Adds a text vocabulary to the {@link BertVocabulary}.
         *
         * @param url the text file url
         * @return this {@code VocabularyBuilder}
         * @throws IOException if failed to read vocabulary file
         */
        public BertVocabulary.Builder addFromTextFile(URL url) throws IOException {
            try (InputStream is = url.openStream()) {
                add(Utils.readLines(is, true));
            }
            return this;
        }

        /**
         * Adds a customized vocabulary to the {@link BertVocabulary}.
         *
         * @param url the text file url
         * @param lambda the function to parse the vocabulary file
         * @return this {@code VocabularyBuilder}
         */
        public BertVocabulary.Builder addFromCustomizedFile(URL url, Function<URL, List<String>> lambda) {
            return add(lambda.apply(url));
        }

        /**
         * Builds the {@link BertVocabulary} object with the set arguments.
         *
         * @return the {@link BertVocabulary} object built
         */
        public BertVocabulary build() {
            return new BertVocabulary(this);
        }
    }

    /**
     * {@code TokenInfo} represents the information stored in the {@link BertVocabulary} about a
     * given token.
     */
    private static final class TokenInfo {
        int frequency;
        long index = -1;

        public TokenInfo() {}
    }

    public Map<String, TokenInfo> getTokenInfoMap(){
        return this.tokens;
    }

    public Map<String, Integer> getTokenMap(){
        Map<String,Integer> tokenMap=new ConcurrentHashMap<>();
        for(String token:this.tokens.keySet()){
            int index= (int) this.tokens.get(token).index;
            tokenMap.put(token,index);
        }
        return tokenMap;
    }
}
