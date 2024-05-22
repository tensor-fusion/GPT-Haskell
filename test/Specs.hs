{-# LANGUAGE OverloadedLists #-}

import Control.Monad.State
import Control.Monad.Trans (liftIO)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T
import Debug.Trace
import Tokenizer
import Model
import Numeric.LinearAlgebra
import System.FilePath ((</>))
import Test.Hspec

modelDir :: FilePath
modelDir = "." </> "models"

modelName :: FilePath
modelName = "124M"

main :: IO ()
main = hspec $ do
  describe "Text encoding and decoding" $ do
    it "encodes text into BPE tokens correctly" $ do
      encoder <- getEncoder modelDir modelName
      result <- evalStateT (encode (T.pack "hello world")) encoder
      liftIO $ result `shouldBe` [31373, 6894]
    it "decodes BPE tokens back to text" $ do
      decoder <- getEncoder modelDir modelName
      result <- evalStateT (decode [31373, 6894]) decoder
      liftIO $ result `shouldBe` T.pack "hello world"

  describe "Numerical activation functions" $ do
    it "correctly computes the GELU activation function" $ do
      let input = matrix 1 [0.5, -0.5, 1.0, 0.0, -1.0] :: Matrix Double
          expected = matrix 1 [0.3457, -0.1543, 0.8412, 0.0, -0.1588] :: Matrix Double
          output = gelu input
          differences = flatten $ cmap abs (output - expected)
      toList differences `shouldSatisfy` all (approx 0.0001)

    it "correctly computes the layer normalization" $ do
      let x = matrix 2 [1.0, 2.0, 3.0, 4.0]
          g = vector [1.0, 1.0]
          b = vector [0.0, 0.0]
          eps = 1e-5
          expected = matrix 2 [-1.0, 1.0, -1.0, 1.0]
          output = layerNorm x g b eps
          differences = flatten $ cmap abs (output - expected)
      toList differences `shouldSatisfy` all (< 0.0001)

  describe "linear function" $ do
    it "correctly performs linear transformation" $ do
      let x = matrix 2 [1.0, 2.0, 3.0, 4.0]
          w = (2 >< 2) [1.0, 0.0, 0.0, 1.0]
          b = vector [0.1, 0.2]
          expected = (2 >< 2) [1.1, 2.2, 3.1, 4.2]
          output = linear x w b
          differences = flatten $ cmap abs (output - expected)
      toList differences `shouldSatisfy` all (< 0.0001)

  describe "Feed-Forward Network (FFN) function" $ do
    it "correctly computes the output of the FFN" $ do
      let x = (2 >< 2) [1.0, 2.0, 3.0, 4.0]
          w1 = (2 >< 8) [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
          b1 = vector [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
          w2 = (8 >< 2) [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
          b2 = vector [0.1, 0.1]
          expected = (2 >< 2) [0.30972894, 0.30972894, 0.60434535, 0.60434535]
          output = ffn x (w1, b1) (w2, b2) -- Output from FFN
          differences = flatten $ cmap abs (output - expected)
      toList differences `shouldSatisfy` all (< 0.0001)

  describe "attention function" $ do
    it "correctly computes attention mechanism" $ do
      let q = matrix 2 [1.0, 2.0, 3.0, 4.0]
          k = (2 >< 2) [1.0, 0.0, 0.0, 1.0]
          v = (2 >< 2) [1.0, 2.0, 3.0, 4.0]
          mask = (2 >< 2) [0.0, -1e10, -1e10, 0.0]
          expected = (2 >< 2) [1.0, 2.0, 3.0, 4.0]
          output = attention q k v mask
          differences = flatten $ cmap abs (output - expected)
      toList differences `shouldSatisfy` all (< 0.0001)

  describe "Multi-Head Attention function" $ do
    it "correctly computes multi-head attention output" $ do
      let x = (2 >< 2) [1.0, 2.0, 3.0, 4.0]
          wAttn = (2 >< 6) $ concat (replicate 2 [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
          bAttn = vector $ replicate 6 0.1
          wProj = (2 >< 2) $ concat (replicate 2 [0.1, 0.1])
          bProj = vector [0.1, 0.1]
          nHead = 2

          output = multiHeadAttentionBlock x ((wAttn, bAttn), (wProj, bProj)) nHead

          expected = (2 >< 2) [0.18000000000000002, 0.18000000000000002, 0.2263459401719, 0.2263459401719]
          differences = flatten $ cmap abs (output - expected)
      toList differences `shouldSatisfy` all (< 0.001)

approx :: Double -> Double -> Bool
approx epsilon x = abs x < epsilon
