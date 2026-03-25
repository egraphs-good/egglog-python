{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE LambdaCase #-}

module Main where

import Data.SRTree
import Data.SRTree.EqSat (simplifyEqSat)
import Data.SRTree.Print (showPython)
import System.CPUTime (getCPUTime)
import System.Environment (getArgs)
import System.Exit (die)

data RunResult = RunResult
  { rowIndex :: Int
  , runtimeSec :: Double
  , stopReason :: String
  , beforeParams :: Int
  , afterParams :: Int
  , beforeNodeCount :: Int
  , afterNodeCount :: Int
  , finalMemoSize :: Maybe Int
  , finalEClassCount :: Maybe Int
  , simplifiedPython :: String
  , notes :: [String]
  }

main :: IO ()
main = do
  args <- getArgs
  let rows = if null args then [1, 50] else map read args
  mapM_ runRow rows

runRow :: Int -> IO ()
runRow row =
  case selectedRowExpr row of
    Nothing -> die $ "helper currently supports rows 1 and 50 only, got " <> show row
    Just tree -> do
      start <- getCPUTime
      let simplified = simplifyEqSat tree
          simplifiedPy = showPython simplified
      simplifiedPy `seq` pure ()
      end <- getCPUTime
      let elapsed = fromIntegral (end - start) / 1e12 :: Double
          result =
            RunResult
              { rowIndex = row
              , runtimeSec = elapsed
              , stopReason = "unavailable_from_exported_api"
              , beforeParams = countParams . fst $ floatConstsToParam tree
              , afterParams = countParams . fst $ floatConstsToParam simplified
              , beforeNodeCount = countNodes tree
              , afterNodeCount = countNodes simplified
              , finalMemoSize = Nothing
              , finalEClassCount = Nothing
              , simplifiedPython = simplifiedPy
              , notes =
                  [ "This helper uses the source repo's exported simplifyEqSat."
                  , "The public API does not expose intermediate e-graph sizes or stop reasons."
                  ]
              }
      printRow result

printRow :: RunResult -> IO ()
printRow res = do
  putStrLn $ "row\t" <> show (rowIndex res)
  putStrLn $ "runtime_sec\t" <> show (runtimeSec res)
  putStrLn $ "stop_reason\t" <> stopReason res
  putStrLn $ "before_params\t" <> show (beforeParams res)
  putStrLn $ "after_params\t" <> show (afterParams res)
  putStrLn $ "before_node_count\t" <> show (beforeNodeCount res)
  putStrLn $ "after_node_count\t" <> show (afterNodeCount res)
  putStrLn $ "final_memo_size\t" <> maybe "na" show (finalMemoSize res)
  putStrLn $ "final_eclass_count\t" <> maybe "na" show (finalEClassCount res)
  mapM_ (\note -> putStrLn $ "note\t" <> note) (notes res)
  putStrLn $ "simplified_python\t" <> simplifiedPython res
  putStrLn "---"

selectedRowExpr :: Int -> Maybe (Fix SRTree)
selectedRowExpr = \case
  1 -> Just row1Expr
  50 -> Just row50Expr
  _ -> Nothing

row1Expr :: Fix SRTree
row1Expr = sqr (-9.29438919215253 + 2.93547417364396 * theta_)

row50Expr :: Fix SRTree
row50Expr =
  ( exp (0.743694003014863 * alpha_)
      * ((-0.0121179632900701 * theta_) + (0.00904122619609017 * alpha_))
      * ((-3.05659895630567 * theta_) + 8.63005732191704)
      + (-0.557193153898209 * alpha_)
      - log (0.782997897866162 * theta_)
      + sqr (exp (-0.144728813168975 * theta_)) * (-1.54770141702422 + (-3.31046821812388 * theta_))
      + (6.34043434659957 * beta_ * 0.643712432648199)
  )
    * (-0.0413897531650583)
    + 0.530747148732844

alpha_, beta_, theta_ :: Fix SRTree
alpha_ = var 0
beta_ = var 1
theta_ = var 2

sqr :: Fix SRTree -> Fix SRTree
sqr x = x ** 2
