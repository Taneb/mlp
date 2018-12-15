{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE PolyKinds #-}
module MLP.Layers.AddBiases where

import Control.DeepSeq
import System.Random.MWC.Distributions

import MLP.Core
import MLP.Classes

-- Add bias layers

data AddBias i o where
  AddBias :: { bias :: Double } -> AddBias ('Tip '[]) ('Tip '[])

instance FeedForwardCPU AddBias where
  feedForwardCPU AddBias {bias} a = a + bias

instance BackPropagateCPU AddBias where
  backPropagateCPU learningRate AddBias {bias} (ia, og) =
    ((ia + bias, og), AddBias {bias = bias + learningRate * og})

instance RandomInitialize AddBias where
  randomInitialize AddBias {} gen = AddBias <$> standard gen 

instance NormalFormNetwork AddBias where
  normalFormNetwork AddBias {bias} = rnf bias

