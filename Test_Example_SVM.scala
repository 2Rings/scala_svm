//package SVM
//
//import org.apache.spark.SparkContext._
//import org.apache.spark.rdd.RDD
//import org.apache.spark.Logging
//import org.apache.spark._
//
//import SVMLabel._
//
//import breeze.linalg.{
//  Matrix => BM,
//  CSCMatrix => BSM,
//  DenseMatrix => BDM,
//  Vector => BV,
//  DenseVector => BDV,
//  SparseVector => BSV,
//  axpy => brzAxpy,
//  svd => brzSvd,
//  max => Bmax,
//  min => Bmin,
//  sum => Bsum
//}
//import scala.collection.mutable.ArrayBuffer
//import scala.collection.mutable.ArrayBuffer
//import java.io._
//
//import scala.math._
//import scala.io._
//import scala.util.Random
//
//object Test_example_SVM {
//  // 构建Spark 对象
//  val conf = new SparkConf().setAppName(" SVMtest")
//  val sc = new SparkContext(conf) 
//  
//  val SVMmodel = new SVMO().setb(0.0).setC(0.1).setEps(0.01).setGamma(1.1).SVMtrain(train_d)
//  
//}