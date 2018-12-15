{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
module MLP.Layers.Sigmoid where

import MLP.Core
import MLP.Classes

-- Sigmoid layer

data Sigmoid i o where
  Sigmoid :: Sigmoid ('Tip '[]) ('Tip '[])

sigmoidFunction :: Double -> Double
sigmoidFunction x = 1/(1 + exp (negate x))

instance FeedForwardCPU Sigmoid where
  feedForwardCPU Sigmoid = sigmoidFunction

instance BackPropagateCPU Sigmoid where
  backPropagateCPU _ Sigmoid (ia, og) =
    let oa = sigmoidFunction ia
        ig = sigmoidFunction ia * (1 - sigmoidFunction ia) * og
    in ((oa, ig), Sigmoid)

instance RandomInitialize Sigmoid where
  randomInitialize Sigmoid _ = pure Sigmoid

instance NormalFormNetwork Sigmoid where
  normalFormNetwork Sigmoid = ()

