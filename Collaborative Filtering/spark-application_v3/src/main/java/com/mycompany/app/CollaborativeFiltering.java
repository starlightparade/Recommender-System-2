package com.mycompany.app;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import org.apache.spark.mllib.recommendation.Rating;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public final class CollaborativeFiltering {
    private static JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("CollaborativeFiltering"));
    private static int K = 1;
    private static final String DATASET_DIR = "reviews_Digital_Music.json";

    public static void main(String[] args) throws Exception {
        JavaRDD<String> data = construct_reviews();

        //split data randomly as training and testing set
        JavaRDD<String>[] split_list = data.randomSplit(new double[]{0.8, 0.2});
        JavaRDD<String> training = split_list[0];
        JavaRDD<String> test = split_list[1];

        JavaRDD<Rating> training_ratings = get_rating(training);
        JavaRDD<Rating> test_ratings = get_rating(test);

        for (K = 1; K <= 5; K++) {
            JavaRDD<Tuple2<Object, Rating[]>> userRecommendationsScaled = get_top_k_recommendations_scaled(training_ratings);

            JavaPairRDD<Object, List<Integer>> userRecommendedList = get_user_recommendation_list(userRecommendationsScaled);

            JavaRDD<Tuple2<Integer, Integer>> ratesAndPreds = get_ratings_predictions(userRecommendedList, test_ratings);

            get_conversion_rate(ratesAndPreds);

            System.out.println("K = " + K + " count = " + data.count());
            System.out.println("K = " + K + " training_count = " + training.count());
            System.out.println("K = " + K + " test_count = " + test.count());
        }

    }

    public static JavaRDD<String> construct_reviews() {
        SparkSession spark = SparkSession
                .builder()
                .getOrCreate();

        Dataset<Row> df = spark.read().json(DATASET_DIR);

        Dataset<Row> doc_df = df.select("reviewerID", "asin", "overall");
        //System.out.println("no. of kindle reviews = " + doc_df.count());

        JavaPairRDD<String, Integer> temp_users = doc_df.toJavaRDD()
                .mapToPair(new PairFunction<Row, String, Integer>() {
                    public Tuple2<String, Integer> call(Row row) throws Exception {
                        return new Tuple2<String, Integer>((String) row.get(0), 1);
                    }
                })
                .reduceByKey((a, b) -> a + b);

        //Filter out user if his/her reviews < 10 in the whole dataset
        List<String> users = temp_users
                .filter(tuple -> !(tuple._2 < 10))
                .map(tuple -> tuple._1)
                .collect();

        Map<String, Integer> reviewerID_map = new HashMap<String, Integer>();
        Map<String, Integer> asin_map = new HashMap<String, Integer>();
        //convert dataset into JavaRDDString(int reviewerID + int asin + double overall)
        JavaRDD<String> data = doc_df.toJavaRDD()
                .filter(new Function<Row, Boolean>() {
                    @Override
                    public Boolean call(Row row) throws Exception {
                        boolean flag = users.contains(row.get(0).toString());
                        return flag;
                    }
                })
                .map(new Function<Row, String>() {
                    public String call(Row row) throws Exception {
                        String reviewerID = row.get(0).toString();
                        String asin = row.get(1).toString();
                        if (!reviewerID_map.keySet().contains(reviewerID)) {
                            int reviewer_ID = reviewerID_map.size();
                            reviewerID_map.put(reviewerID, reviewerID_map.size());
                            reviewerID = String.valueOf(reviewer_ID);
                        } else {
                            reviewerID = String.valueOf(reviewerID_map.get(reviewerID));
                        }

                        if (!asin_map.keySet().contains(asin)) {
                            int asin_ID = asin_map.size();
                            asin_map.put(asin, asin_map.size());
                            asin = String.valueOf(asin_ID);
                        } else {
                            asin = String.valueOf(asin_map.get(asin));
                        }

                        String review = reviewerID + ',' + asin + ',' + row.get(2).toString();
                        return review;
                    }
                });

        return data;
    }

    // Convert rating from String to Rating Object
    public static JavaRDD<Rating> get_rating(JavaRDD<String> data) {
        JavaRDD<Rating> ratings = data.map(s -> {
            String[] sarray = s.split(",");
            return new Rating(Integer.parseInt(sarray[0]),
                    Integer.parseInt(sarray[1]),
                    Double.parseDouble(sarray[2]));
        });
        return ratings;
    }

    public static JavaRDD<Tuple2<Object, Rating[]>> get_top_k_recommendations_scaled(JavaRDD<Rating> ratings) {
        // Build the recommendation model using ALS
        int rank = 10;
        int numIterations = 10;
        MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);

        // Get top K recommendations for every user and scale ratings from 0 to 1
        JavaRDD<Tuple2<Object, Rating[]>> userRecommendations = model.recommendProductsForUsers(K).toJavaRDD();
        JavaRDD<Tuple2<Object, Rating[]>> userRecommendationsScaled = userRecommendations.map(
                new Function<Tuple2<Object, Rating[]>, Tuple2<Object, Rating[]>>() {
                    public Tuple2<Object, Rating[]> call(Tuple2<Object, Rating[]> t) {
                        Rating[] scaledRatings = new Rating[t._2().length];
                        for (int i = 0; i < scaledRatings.length; i++) {
                            double newRating = Math.max(Math.min(t._2()[i].rating(), 1.0), 0.0);
                            scaledRatings[i] = new Rating(t._2()[i].user(), t._2()[i].product(), newRating);
                        }
                        return new Tuple2<Object, Rating[]>(t._1(), scaledRatings);
                    }
                }
        );
        userRecommendationsScaled.saveAsTextFile("userRecsScaled"+K);
        return userRecommendationsScaled;
    }

    public static JavaPairRDD<Object, List<Integer>> get_user_recommendation_list(JavaRDD<Tuple2<Object, Rating[]>> userRecommendationsScaled) {
        JavaPairRDD<Object, Rating[]> userRecommended = JavaPairRDD.fromJavaRDD(userRecommendationsScaled);

        // Extract the product id from each recommendation
        JavaPairRDD<Object, List<Integer>> userRecommendedList = userRecommended.mapValues(
                new Function<Rating[], List<Integer>>() {
                    public List<Integer> call(Rating[] docs) {
                        List<Integer> products = new ArrayList<Integer>();
                        for (Rating r : docs) {
                            products.add(r.product());
                        }
                        return products;
                    }
                }
        );
        System.out.println("K = " + K + " userRecommendedList.count() = " + userRecommendedList.count());
        userRecommendedList.saveAsTextFile("userRecommendedList"+K);
        return userRecommendedList;
    }

    public static JavaRDD<Tuple2<Integer, Integer>> get_ratings_predictions(JavaPairRDD<Object, List<Integer>> userRecommendedList, JavaRDD<Rating> test_ratings) {
        List<JavaPairRDD<Object, List<Integer>>> userRecommended_list = new ArrayList<JavaPairRDD<Object, List<Integer>>>();
        userRecommended_list.add(userRecommendedList);

        Map<Object, List<Integer>> userRecommended_map = userRecommendedList.collectAsMap();

        List<Rating> test_ratings_list = test_ratings.collect();
        Map<Integer, Rating> map_temp = new HashMap<>();
        for (Rating rating : test_ratings_list){
            map_temp.put(rating.user(), rating);
        }

        JavaRDD<Tuple2<Integer, Integer>> ratesAndPreds = userRecommendedList
                .map(tuple -> {
                    int user_id = (int)tuple._1;
                    List<Integer> recom_products = tuple._2;
                    int conversionRate = 0;

                    if (map_temp.keySet().contains(user_id)){
                        Rating rating = map_temp.get(user_id);
                        if(recom_products.contains(rating.product())){
                            conversionRate = 1;
                        }
                    }

                    return new Tuple2<Integer, Integer>(user_id, conversionRate);
                });
        ratesAndPreds.saveAsTextFile("ratesAndPreds"+K);
        return ratesAndPreds;
    }

    public static JavaPairRDD<Integer, Integer> get_conversion_rate(JavaRDD<Tuple2<Integer, Integer>> ratesAndPreds) {
        JavaPairRDD<Integer, Integer> conversionRate = ratesAndPreds
                .mapToPair(tuple -> {
                    return new Tuple2<Integer, Integer>(tuple._2, 1);
                })
                .reduceByKey((a, b) -> a + b);
        conversionRate.saveAsTextFile("conversionRate"+K);
        return conversionRate;
    }

}
