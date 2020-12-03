-- | PPL contains three statements:
--
-- sample :: Address -> Dist -> PPL Double
-- - sample takes an address (which is a string), a distribution object and returns a float
--
-- observe :: Address -> Dist -> Double -> PPL ()
-- - observe takes an address, dist, and observation and alters state
--
-- end :: [Double] -> PPL [Double]
-- - end terminates the program and returns anything you'd like to return

{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE NoImplicitPrelude #-}
module Main where

import Prelude hiding (putStrLn)
import qualified Prelude
import Control.Monad.Free
import qualified System.Random.MWC.Probability as MWC
import Control.Monad.State

import Lib

-- Assuming we have some kind of black box that can produce floats in a pure-way
type BlackBox = Double -> Double -> Double

-- we can write a program, like on page 226, as follows:
programOnPage226 :: [Double] -> BlackBox -> BlackBox -> Double -> PPL [Double]
programOnPage226 y ηy ηv θ = do
  z' <- sample "z" (Multinomial 1 [θ])
  let z = head z'
  v' <- sample "v" (Normal (ηy z θ) (ηv z θ))
  let v = head v'
  observe "y" (Normal (ηy v θ) 1) y
  return [z, v]


main :: IO ()
main = do
  s <- MWC.createSystemSeed
  printPPL $ program
  print $ evalState (stateHandler program) (s, mempty)
  where
    program = programOnPage226 [1] (+) (-) 2

