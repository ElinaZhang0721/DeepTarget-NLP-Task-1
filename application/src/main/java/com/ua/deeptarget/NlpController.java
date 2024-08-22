package com.ua.deeptarget;

import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;

import java.io.InputStream;
import java.io.FileNotFoundException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Properties;
import java.math.BigDecimal;
import java.math.RoundingMode;

@Controller
public class NlpController {

    private static final Logger logger = LoggerFactory.getLogger(NlpController.class);

    @GetMapping("/input")
    public String showInputForm(Model model) {
        return "input";
    }

    @PostMapping("/analyze")
    public String analyzeText(@RequestParam(value = "text", required = false) String text,
                              @RequestParam("file") MultipartFile file,
                              Model model) throws Exception {

        // Check if no text or file is provided
        if ((text == null || text.isEmpty()) && file.isEmpty()) {
            model.addAttribute("error", "Please provide text input or upload a file.");
            return "input"; 
        }

        // If text input is provided, use it; otherwise, read from the file
        if (text != null && !text.isEmpty()) {
            logger.info("Processing text input.");
        } else if (!file.isEmpty()) {
            logger.info("Processing file upload.");
            text = new String(file.getBytes(), StandardCharsets.UTF_8);
        }

        // Perform NLP analysis
        Map<String, Object> nlpStatistics = performNlpAnalysis(text);

        // Add the analysis results to the model and show the results view
        model.addAttribute("nlpStatistics", nlpStatistics);

        return "results";
    }

    private Map<String, Object> performNlpAnalysis(String text) throws Exception {
        Map<String, Object> statistics = new LinkedHashMap<>();
    
        // Sentence Count
        int sentenceCount = 0;
        try (InputStream sentenceModelStream = getClass().getResourceAsStream("/opennlp-models/en-sent.bin")) {
            if (sentenceModelStream == null) {
                throw new FileNotFoundException("Sentence model 'en-sent.bin' not found in resources.");
            }
            // Load sentence model and detect sentences
            SentenceModel sentenceModel = new SentenceModel(sentenceModelStream);
            SentenceDetectorME sentenceDetector = new SentenceDetectorME(sentenceModel);
            sentenceCount = sentenceDetector.sentDetect(text).length;

            // Store sentence count in statistics
            statistics.put("Sentence Count", sentenceCount);

            // Compare sentenceCount with the aggregated result
            int sentContdiff = sentenceCount - 32;

            String sentenceCountComparison;
            if (sentContdiff < 0) {
                sentenceCountComparison = "Noun count is lower than Aggregated Result by " + Math.abs(sentContdiff);
            } else if (sentContdiff > 0) {
                sentenceCountComparison = "Noun count is higher than Aggregated Result by " + sentContdiff;
            } else {
                sentenceCountComparison = "Equal to Aggregated Result";
            }

            // Store the comparison result in statistics
            statistics.put("Sentence Count Comparison", sentenceCountComparison);
    
        } catch (Exception e) {
        logger.error("Error loading sentence model or detecting sentences", e);
        statistics.put("Sentence Count", 0);
        statistics.put("Sentence Count Comparison", "Error in sentence detection");
        }
    
        // Word Count
        SimpleTokenizer tokenizer = SimpleTokenizer.INSTANCE;
        String[] tokens = tokenizer.tokenize(text);
        int wordCount = tokens.length;

        // Compare wordCount with Aggregated Results 750
        int wordCountdiff = wordCount - 750;

        String wordCountComparison;
        if (wordCountdiff > 0) { 
            wordCountComparison = "Word Count is higher than Aggregated Result by " + wordCountdiff;
        } else if (wordCountdiff < 0) {
            wordCountComparison = "Word Count is lower than Aggregated Result by " + Math.abs(wordCountdiff);
        } else {
            wordCountComparison = "Equal to Aggregated Result";
        }

        // Store the results in the statistics map
        statistics.put("Word Count", wordCount);
        statistics.put("Word Count Comparison", wordCountComparison);
    
        // Noun Count
        try (InputStream posModelStream = getClass().getResourceAsStream("/opennlp-models/en-pos-maxent.bin")) {
            if (posModelStream == null) {
                throw new FileNotFoundException("POS model 'en-pos-maxent.bin' not found in resources.");
            }
            POSModel posModel = new POSModel(posModelStream);
            POSTaggerME posTagger = new POSTaggerME(posModel);
            String[] posTags = posTagger.tag(tokens);
    
            int nounCount = 0;
            for (String posTag : posTags) {
                if (posTag.equals("NN") || posTag.equals("NNS") || posTag.equals("NNP") || posTag.equals("NNPS")) {
                    nounCount++;
                }
            }

            // Calculate the difference between nounCount and 322
            int nounCountdiff = nounCount - 322;

            String nounCountComparison;
            if (nounCountdiff < 0) {
                nounCountComparison = "Noun count is lower than Aggregated Result by " + Math.abs(nounCountdiff);
            } else if (nounCountdiff > 0) {
                nounCountComparison = "Noun count is higher than Aggregated Result by " + nounCountdiff;
            } else {
                nounCountComparison = "Equal to Aggregated Result";
            }

            statistics.put("Noun Count", nounCount);
            statistics.put("Noun Count Comparison", nounCountComparison);
        } catch (Exception e) {
            logger.error("Error loading POS model or detecting nouns", e);
            statistics.put("Noun Count", 0);
        }
    
        // Sentiment Analysis using Stanford CoreNLP
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
    
            // Create an annotation for the input text
            Annotation annotation = new Annotation(text);
            pipeline.annotate(annotation);
    
            double totalSentimentScore = 0.0;
            int numSentences = 0;
    
            // Loop through each sentence and calculate sentiment
            for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
            Tree sentimentTree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
            int sentimentScore = RNNCoreAnnotations.getPredictedClass(sentimentTree);
    
            double sentimentValue = mapSentimentToRange(sentimentScore);
            totalSentimentScore += sentimentValue;
    
            String sentimentLabel = mapSentimentToLabel(sentimentValue);
            statistics.put("Sentiment for Sentence " + (++numSentences), sentimentLabel + " (" + sentimentValue + ")");
            }
    
        // Calculate Overall Sentiment
        double averageSentiment = (numSentences > 0) ? (totalSentimentScore / numSentences) : 0.0;
        String overallSentimentLabel = mapSentimentToLabel(averageSentiment);

        // Use BigDecimal to round the averageSentiment to 4 decimal places
        BigDecimal bd = new BigDecimal(averageSentiment);
        bd = bd.setScale(4, RoundingMode.HALF_UP);
        averageSentiment = bd.doubleValue();

        statistics.put("Overall Sentiment", overallSentimentLabel + " (" + bd.toPlainString() + ")");

        // Compare with Aggregated Result

        BigDecimal aggregatedSentiment = new BigDecimal("0.0691");
        BigDecimal overallSentidiff = bd.subtract(aggregatedSentiment);

        String overallSentiCompare;
        if (overallSentidiff.compareTo(BigDecimal.ZERO) < 0) {
            overallSentiCompare = "Overall Sentiment is lower than Aggregated Result by " + overallSentidiff.abs().toPlainString();
        } else if (overallSentidiff.compareTo(BigDecimal.ZERO) > 0) {
            overallSentiCompare = "Overall Sentiment is higher than Aggregated Result by " + overallSentidiff.toPlainString();
        } else {
            overallSentiCompare = "Equal to Aggregated Result";
        }

        statistics.put("Overall Sentiment Comparison", overallSentiCompare);
    
        // Flesch-Kincaid Readability Score Calculation
        int totalSyllables = calculateSyllableCount(text);
        double fleschKincaidScore = calculateFleschKincaidScore(wordCount, sentenceCount, totalSyllables);
        BigDecimal bdScore = new BigDecimal(fleschKincaidScore).setScale(4, RoundingMode.HALF_UP);
        String readabilityLevel = interpretFleschKincaidScore(fleschKincaidScore);
    
        // Add readability score and interpretation to statistics
        statistics.put("Flesch-Kincaid Readability Score", bdScore.toString());
        statistics.put("Readability Level", readabilityLevel);

        // Compare with Aggerated Result of Readability Score Comparison
        BigDecimal aggregatedScore = new BigDecimal("58.1533");
        BigDecimal readscorediff = bdScore.subtract(aggregatedScore);

        String readscoreCompare;
        if (readscorediff.compareTo(BigDecimal.ZERO) < 0) {
            readscoreCompare = "Readability score is lower than Aggregated Result by " + readscorediff.abs();
        } else if (readscorediff.compareTo(BigDecimal.ZERO) > 0) {
            readscoreCompare = "Readability score is higher than Aggregated Result by " + readscorediff;
        } else {
        readscoreCompare = "Equal to Aggregated Result";
        }

        statistics.put("Readability Score Comparison", readscoreCompare);

        // Compare with Aggerated Result of Readability Level Comparison
        String readlevelCompare;
        if (readscorediff.compareTo(BigDecimal.ZERO) < 0) {
            readlevelCompare = "Easier to read";
        } else if (readscorediff.compareTo(BigDecimal.ZERO) > 0) {
            readlevelCompare = "Harder to read";
        } else {
            readlevelCompare = "Equal Read Level";
        }

        statistics.put("Readability Level Comparison", readlevelCompare);        
        
    
        return statistics;
    }

    // Calculate total syllable count for the text
    private int calculateSyllableCount(String text) {
        int syllableCount = 0;
        String[] words = text.split("\\s+");
        for (String word : words) {
            syllableCount += countSyllables(word);
        }
        return syllableCount;
    }

    // Helper method to count syllables in a word (approximation)
    private int countSyllables(String word) {
        word = word.toLowerCase();
        String vowels = "aeiouy";
        boolean lastWasVowel = false;
        int syllableCount = 0;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (vowels.indexOf(c) >= 0) {
                if (!lastWasVowel) {
                    syllableCount++;
                    lastWasVowel = true;
                }
            } else {
                lastWasVowel = false;
            }
        }

        // Adjust for silent 'e'
        if (word.endsWith("e") && syllableCount > 1) {
            syllableCount--;
        }

        // Ensure at least one syllable per word
        return syllableCount > 0 ? syllableCount : 1;
    }

    // Flesch-Kincaid readability score calculation in Flesch-Kincaid reading ease formula
    private double calculateFleschKincaidScore(int wordCount, int sentenceCount, int syllableCount) {
        if (sentenceCount == 0 || wordCount == 0) {
            return 0.0; // Prevent division by zero
        }
        return 206.835 - 1.015 * (double) wordCount / sentenceCount - 84.6 * (double) syllableCount / wordCount;
    }

    // Interpret the Flesch-Kincaid score into readability levels
    private String interpretFleschKincaidScore(double score) {
        if (score >= 90) {
            return "Very easy to read";
        } else if (score >= 60) {
            return "Easily understood by 13-15-year-olds";
        } else if (score >= 30) {
            return "College-level text";
        } else {
            return "Very difficult to read";
        }
    }

    // Map sentiment scores to the range of -1 to 1
    private double mapSentimentToRange(int sentimentScore) {
        switch (sentimentScore) {
            case 0: // Very Negative
                return -1.0;
            case 1: // Negative
                return -0.5;
            case 2: // Neutral
                return 0.0;
            case 3: // Positive
                return 0.5;
            case 4: // Very Positive
                return 1.0;
            default:
                return 0.0; // Default to neutral if unknown sentiment
        }
    }

    // Map sentiment values to human-readable labels
    private String mapSentimentToLabel(double sentimentValue) {
        if (sentimentValue >= 0.5) {
            return "Positive";
        } else if (sentimentValue > -0.5 && sentimentValue < 0.5) {
            return "Neutral";
        } else {
            return "Negative";
        }
    }
}
