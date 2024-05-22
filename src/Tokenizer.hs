{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

module Tokenizer where

import Control.Monad.State
import Data.Aeson
import qualified Data.ByteString as B
import Data.Char (chr, ord)
import Data.List (foldl', minimumBy)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe, mapMaybe)
import Data.Ord (compare, comparing)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import Data.Text.Encoding
import qualified Data.Text.Encoding as E
import Data.Tuple (swap)
import Debug.Trace
import System.FilePath ((</>))
import Text.Printf
import Text.Regex.TDFA (AllTextMatches (getAllTextMatches), (=~))
import qualified Text.Regex.TDFA as PCRE

type ByteEncoder = Map.Map Int Char

type ByteDecoder = Map.Map Char Int

type BpeRanks = Map.Map (Text, Text) Int

type Cache = Map.Map Text Text

data Encoder = Encoder
  { encoder :: Map.Map Text Int,
    decoder :: Map.Map Int Text,
    byteEncoder :: ByteEncoder,
    byteDecoder :: ByteDecoder,
    bpeRanks :: BpeRanks,
    cache :: Cache
  }

generateMappings :: [(Int, Int)]
generateMappings =
  zip [ord '!' .. ord '~'] [ord '!' .. ord '~']
    ++ zip [ord '¡' .. ord '¬'] [ord '¡' .. ord '¬']
    ++ zip [ord '®' .. ord 'ÿ'] [ord '®' .. ord 'ÿ']

bytesToUnicode :: (ByteEncoder, ByteDecoder)
bytesToUnicode =
  let fullRange = [0 .. 255]
      bs = map fst generateMappings
      n = length bs

      extendMappings :: [(Int, Int)] -> Int -> [(Int, Int)]
      extendMappings mappings next =
        foldl'
          ( \acc b ->
              if b `notElem` bs
                then acc ++ [(b, 256 + next + b - n)]
                else acc
          )
          mappings
          fullRange

      initialMappings = generateMappings
      finalMappings = extendMappings initialMappings 0
      byteEncoder = Map.fromList [(b, chr c) | (b, c) <- finalMappings]
      byteDecoder = Map.fromList [(chr c, b) | (b, c) <- finalMappings]
   in (byteEncoder, byteDecoder)

(byteEncoderInit, byteDecoderInit) = bytesToUnicode

getPairs :: [Text] -> Set.Set (Text, Text)
getPairs word =
  let createPairs :: [Text] -> [(Text, Text)]
      createPairs [] = []
      createPairs [_] = []
      createPairs (x : y : xs) = (x, y) : createPairs (y : xs)
   in Set.fromList (createPairs word)

initialByteEncoder :: ByteEncoder
initialByteEncoder = Map.empty

initialByteDecoder :: ByteDecoder
initialByteDecoder = Map.empty

type EncoderState a = StateT Encoder IO a

textToTextList :: Text -> [Text]
textToTextList = map T.singleton . T.unpack

textListToText :: [Text] -> Text
textListToText = T.concat

bpe :: Text -> EncoderState Text
bpe token = do
  enc <- get
  case Map.lookup token (cache enc) of
    Just result -> return result
    Nothing -> do
      let word = textToTextList token
          pairs = getPairs word
      if Set.null pairs
        then return token
        else do
          finalWord <- debugProcessWord enc word
          let finalToken = textListToText finalWord
          modify (\e -> e {cache = Map.insert token finalToken (cache e)})
          return finalToken

-- processWord :: Encoder -> [Text] -> EncoderState [Text]
-- processWord enc word =
--   let pairs = getPairs word
--    in if Set.null pairs
--         then return word
--         else do
--           let bigram = minimumBy (comparing (\p -> Map.findWithDefault maxBound p (bpeRanks enc))) (Set.toList pairs)
--           if not (bigram `Map.member` bpeRanks enc)
--             then return word
--             else do
--               let newWord = mergePairs word bigram
--               processWord enc newWord

debugProcessWord :: Encoder -> [Text] -> EncoderState [Text]
debugProcessWord enc word = do
  let pairs = getPairs word
  liftIO $ putStrLn $ "Current word: " ++ show word
  liftIO $ putStrLn $ "Current pairs: " ++ show pairs
  if Set.null pairs
    then return word
    else do
      let bigram = minimumBy (comparing (\p -> Map.findWithDefault maxBound p (bpeRanks enc))) (Set.toList pairs)
      liftIO $ putStrLn $ "Selected pair to merge: " ++ show bigram
      if not (bigram `Map.member` bpeRanks enc)
        then return word
        else do
          let newWord = mergePairs word bigram
          debugProcessWord enc newWord

mergePairs :: [Text] -> (Text, Text) -> [Text]
mergePairs [] _ = []
mergePairs [x] _ = [x]
mergePairs (x : y : xs) (first, second)
  | x == first && y == second = (x <> y) : mergePairs xs (first, second)
  | otherwise = x : mergePairs (y : xs) (first, second)

pat :: Text
pat = "('s|'t|'re|'ve|'m|'ll|'d|\\p{L}+|\\p{N}+|[^\\s\\p{L}\\p{N}]+|\\s+)"

compiledPat :: PCRE.Regex
compiledPat = PCRE.makeRegex (T.unpack pat)

encode :: Text -> EncoderState [Int]
encode text = do
  enc <- get
  let tokens = map T.pack $ getAllTextMatches $ T.unpack text =~ T.unpack pat :: [Text]
  liftIO $ putStrLn $ "Tokens: " ++ show tokens
  bpeTokens <- mapM bpe tokens
  let encodedTokens = [fromMaybe (-1) (Map.lookup t (encoder enc)) | t <- T.words $ T.unwords bpeTokens]
  liftIO $ putStrLn $ "Encoded Tokens: " ++ show encodedTokens
  return encodedTokens

processToken :: Encoder -> Text -> EncoderState [Int]
processToken enc token = do
  let byteToken = T.concat [T.singleton $ fromMaybe '?' (Map.lookup (fromIntegral $ B.head $ E.encodeUtf8 $ T.singleton c) (byteEncoder enc)) | c <- T.unpack token]
  bpeResult <- bpe byteToken
  liftIO $ putStrLn $ "processTokenBPE Result for " ++ T.unpack token ++ ": " ++ T.unpack bpeResult
  return [fromMaybe (-1) (Map.lookup t (encoder enc)) | t <- T.words bpeResult]

decode :: [Int] -> EncoderState Text
decode tokens = do
  enc <- get
  let decodedStrings = map (\token -> fromMaybe "" (Map.lookup token (decoder enc))) tokens
  return $ T.unwords decodedStrings

getEncoder :: FilePath -> FilePath -> IO Encoder
getEncoder modelDir modelName = do
  let encoderPath = modelDir </> modelName </> "encoder.json"
      bpePath = modelDir </> modelName </> "vocab.bpe"

  encoderJSON <- readFile encoderPath
  let encoder = decodeStrict' (encodeUtf8 $ T.pack encoderJSON) :: Maybe (Map.Map Text Int)

  bpeData <- readFile bpePath
  let bpeMerges = map ((\[a, b] -> (a, b)) . T.splitOn " ") . tail . init . T.splitOn "\n" $ T.pack bpeData

  case encoder of
    Just enc ->
      return $
        Encoder
          { encoder = enc,
            decoder = Map.fromList $ map swap $ Map.toList enc,
            bpeRanks = Map.fromList $ zip bpeMerges [0 ..],
            cache = Map.empty,
            byteEncoder = byteEncoderInit,
            byteDecoder = byteDecoderInit
          }
    Nothing -> error "Failed to decode encoder.json"