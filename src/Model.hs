{-# LANGUAGE FlexibleContexts #-}

module Model where

import Data.List (concat, foldl', transpose, zip4)
import qualified Data.Vector.Storable as V
import Debug.Trace
import Numeric.LinearAlgebra
  ( Indexable ((!)),
    Konst (konst),
    Matrix,
    Transposable (tr),
    Vector,
    asColumn,
    asRow,
    cmap,
    cols,
    flatten,
    fromRows,
    matrix,
    maxElement,
    maxIndex,
    repmat,
    rows,
    scalar,
    size,
    subMatrix,
    sumElements,
    toList,
    toRows,
    vector,
    (<>),
    (|||),
  )

data LayerNormParams = LayerNormParams
  { gamma :: Vector Double,
    beta :: Vector Double,
    epsilon :: Double
  }

data WeightsBiases = WeightsBiases
  { weights :: Matrix Double,
    biases :: Vector Double
  }

data AttnWeights = AttnWeights
  { attnWeights :: WeightsBiases,
    projWeights :: WeightsBiases
  }

data GPT2Config = GPT2Config
  { tokenEmbeddings :: Matrix Double,
    posEmbeddings :: Matrix Double,
    transformerBlocks :: [TransformerBlockConfig],
    finalLayerNormParams :: LayerNormParams,
    heads :: Int
  }

data TransformerBlockConfig = TransformerBlockConfig
  { mlpWeights :: [WeightsBiases],
    attentionWeights :: AttnWeights,
    ln1Params :: LayerNormParams,
    ln2Params :: LayerNormParams,
    numHeads :: Int
  }

gelu :: Matrix Double -> Matrix Double
gelu = cmap (\z -> 0.5 * z * (1 + tanh (sqrt (2 / pi) * (z + 0.044715 * z ** 3))))

traceMatrix :: String -> Matrix Double -> Matrix Double
traceMatrix msg mat = trace (msg ++ ": " ++ show mat) mat

roundTo :: Int -> Double -> Double
roundTo n f = fromInteger (round $ f * (10 ^ n)) / (10.0 ^^ n)

layerNorm :: Matrix Double -> Vector Double -> Vector Double -> Double -> Matrix Double
layerNorm x g b eps = traceMatrix "Final output" $ cmap (roundTo 4) $ scaleAndShift (normalize (tr x))
  where
    mean = rowMeans (tr x)
    variance = rowVariances (tr x) mean eps
    normalize y =
      (y - repmat (asColumn mean) 1 (cols y))
        / cmap sqrt (repmat (asColumn variance) 1 (cols y))
    scaleAndShift y =
      (y * repmat (asRow g) (rows y) 1)
        + repmat (asRow b) (rows y) 1

rowMeans :: Matrix Double -> Vector Double
rowMeans m = vector (map meanElement (toRows m))

meanElement :: Vector Double -> Double
meanElement v = sumElements v / fromIntegral (size v)

rowVariances :: Matrix Double -> Vector Double -> Double -> Vector Double
rowVariances m means eps = vector (zipWith varElement (toRows m) (toList means))
  where
    varElement row mean = (sumElements (cmap (\x -> (x - mean) ^ 2) row) / fromIntegral (size row)) + eps

linear :: Matrix Double -> Matrix Double -> Vector Double -> Matrix Double
linear x w b = (x Numeric.LinearAlgebra.<> w) + asRow b

softmax :: Matrix Double -> Matrix Double
softmax m =
  let expm = cmap exp (m - konst (maxElement m) (rows m, cols m))
      sumExp = asColumn $ vector $ map sumElements $ toRows expm
      result = expm / sumExp
   in traceMatrix "Softmax output" result

attention :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
attention q k v mask =
  let d_k = fromIntegral (cols q)
      qk = q Numeric.LinearAlgebra.<> tr k
      scaledQK = (qk / scalar (sqrt d_k)) + mask
      weights = softmax (traceMatrix "Scaled attention logits" scaledQK)
      result = weights Numeric.LinearAlgebra.<> v
   in traceMatrix "Attention output" result

ffn :: Matrix Double -> (Matrix Double, Vector Double) -> (Matrix Double, Vector Double) -> Matrix Double
ffn x (w1, b1) (w2, b2) =
  let a = gelu (linear x w1 b1)
      result = linear a w2 b2
   in traceMatrix "ffn output" result

splitAtColumns :: Int -> Matrix Double -> (Matrix Double, Matrix Double)
splitAtColumns idx mat =
  let left = subMatrix (0, 0) (rows mat, idx) mat
      right = subMatrix (0, idx) (rows mat, cols mat - idx) mat
   in (left, right)

splitQKV :: Matrix Double -> (Matrix Double, Matrix Double, Matrix Double)
splitQKV x =
  let (q, rest) = splitAtColumns (cols x `div` 3) x
      (k, v) = splitAtColumns (cols rest `div` 2) rest
   in (q, k, v)

splitHeads :: Int -> Matrix Double -> [Matrix Double]
splitHeads nHead mat = map (\i -> subMatrix (0, i * (cols mat `div` nHead)) (rows mat, cols mat `div` nHead) mat) [0 .. nHead - 1]

mask :: Int -> Matrix Double
mask size = matrix size $ concat [if i < j then [-1e10] else [0] | i <- [0 .. size - 1], j <- [0 .. size - 1]]

concatHeads :: [Matrix Double] -> Matrix Double
concatHeads = foldr1 (|||)

outputProjection :: Matrix Double -> (Matrix Double, Vector Double) -> Matrix Double
outputProjection x (w, b) = linear x w b

multiHeadAttentionBlock :: Matrix Double -> ((Matrix Double, Vector Double), (Matrix Double, Vector Double)) -> Int -> Matrix Double
multiHeadAttentionBlock x ((attn_weights, attn_biases), (proj_weights, proj_biases)) nHead =
  let projected = linear x attn_weights attn_biases
      (q, k, v) = splitQKV projected
      heads = map (\(q', k', v') -> attention q' k' v' (mask (rows q'))) $ zip3 (splitHeads nHead q) (splitHeads nHead k) (splitHeads nHead v)
      concatenated = concatHeads heads
   in outputProjection concatenated (proj_weights, proj_biases)

transformerBlock :: Matrix Double -> TransformerBlockConfig -> Matrix Double
transformerBlock input config = output
  where
    ln1Gamma = gamma (ln1Params config)
    ln1Beta = beta (ln1Params config)
    ln1Epsilon = epsilon (ln1Params config)

    ln2Gamma = gamma (ln2Params config)
    ln2Beta = beta (ln2Params config)
    ln2Epsilon = epsilon (ln2Params config)

    ln1 = layerNorm input ln1Gamma ln1Beta ln1Epsilon

    attnW = weights (attnWeights (attentionWeights config))
    attnB = biases (attnWeights (attentionWeights config))
    projW = weights (projWeights (attentionWeights config))
    projB = biases (projWeights (attentionWeights config))

    mhaOutput = multiHeadAttentionBlock ln1 ((attnW, attnB), (projW, projB)) (numHeads config)
    attnOutput = ln1 + mhaOutput

    ln2 = layerNorm attnOutput ln2Gamma ln2Beta ln2Epsilon

    ffnW1 = weights (head (mlpWeights config))
    ffnB1 = biases (head (mlpWeights config))
    ffnW2 = weights (last (mlpWeights config))
    ffnB2 = biases (last (mlpWeights config))

    ffnOutput = ffn ln2 (ffnW1, ffnB1) (ffnW2, ffnB2)
    output = attnOutput + ffnOutput

gpt2 :: Vector Int -> GPT2Config -> Matrix Double
gpt2 inputs config = output
  where
    tokenEmbeds = fromRows [tokenEmbeddings config ! idx | idx <- toList inputs]
    posEmbeds = subMatrix (0, 0) (rows tokenEmbeds, V.length inputs) (posEmbeddings config)
    x0 = tokenEmbeds + posEmbeds
    xFinal = foldl' applyTransformerBlock x0 (transformerBlocks config)
    normX = layerNorm xFinal (gamma (finalLayerNormParams config)) (beta (finalLayerNormParams config)) (epsilon (finalLayerNormParams config))

    output = normX Numeric.LinearAlgebra.<> tr (tokenEmbeddings config)

    applyTransformerBlock :: Matrix Double -> TransformerBlockConfig -> Matrix Double
    applyTransformerBlock = transformerBlock

generate :: V.Vector Int -> GPT2Config -> Int -> Int -> V.Vector Int
generate inputs config nHead nTokensToGenerate = generate' inputs nTokensToGenerate
  where
    generate' :: V.Vector Int -> Int -> V.Vector Int
    generate' inputsLeft 0 = V.slice (V.length inputsLeft - nTokensToGenerate) nTokensToGenerate inputsLeft
    generate' inputsLeft count =
      let logits = gpt2 inputsLeft config
          numRows = rows logits
          lastLogits = flatten $ subMatrix (numRows - 1, 0) (1, cols logits) logits

          nextId = maxIndex lastLogits

          newInputs = V.snoc inputsLeft nextId
       in generate' newInputs (count - 1)
