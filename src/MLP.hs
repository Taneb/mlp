{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
module MLP where

import Control.Applicative
import Control.Arrow (second)
import Control.Category
import Control.DeepSeq
import Control.Monad.Primitive
import Data.List (foldl')
import GHC.TypeNats
import Linear
import Linear.V
import System.Random.MWC
import System.Random.MWC.Distributions

import Prelude hiding ((.), id)

infixr 1 `Cons`

data Network c i o where
  Nil :: Network c x x
  Cons :: c l => l i m -> Network c m o -> Network c i o

instance Category (Network c) where
  id = Nil
  x . Nil = x
  x . Cons l y = Cons l (x . y)

forget :: c l => l i o -> Network c i o
forget l = Cons l Nil

free :: Category k => (forall l i o. c l => l i o -> k i o) -> Network c i o -> k i o
free f Nil = id
free f (Cons l n) = free f n . f l

class FeedForwardCPU l where
  feedForwardCPU :: l i o -> V i Double -> V o Double

newtype FFCPU i o = FFCPU {runFFCPU :: V i Double -> V o Double}

instance Category FFCPU where
  id = FFCPU id
  FFCPU x . FFCPU y = FFCPU (x . y)

instance (forall l. c l => FeedForwardCPU l) => FeedForwardCPU (Network c) where
  feedForwardCPU = runFFCPU . free (FFCPU . feedForwardCPU)

class FeedForwardCPU l => BackPropagateCPU l where
  backPropagateCPU :: Double -> l i o -> (V i Double, V o Double) -> ((V o Double, V i Double), l i o)

newtype BPCPU c i o = BPCPU {runBPCPU :: (V i Double, V o Double) -> ((V o Double, V i Double), Network c i o)}

instance Category (BPCPU c) where
  id = BPCPU $ \(i, o) -> ((o, i), id)
  BPCPU f . BPCPU g = BPCPU $ \(ia, og) ->
    let ((ma, ig), g') = g (ia, mg)
        ((oa, mg), f') = f (ma, og)
    in ((oa, ig), f' . g')

instance (forall l. c l => BackPropagateCPU l) => BackPropagateCPU (Network c) where
  backPropagateCPU learningRate = runBPCPU . free (\l -> BPCPU $ second forget . backPropagateCPU learningRate l)

class RandomInitialize l where
  randomInitialize :: PrimMonad m => l i o -> Gen (PrimState m) -> m (l i o)

newtype RI m c i o = RI {runRI :: Gen (PrimState m) -> m (Network c i o)}

instance Applicative m => Category (RI m c) where
  id = RI $ \_ -> pure id
  RI f . RI g = RI $ (liftA2 . liftA2) (.) f g

instance (forall l. c l => RandomInitialize l) => RandomInitialize (Network c) where
  randomInitialize = runRI . free (RI . fmap (fmap forget) . randomInitialize)

class NormalFormNetwork l where
  normalFormNetwork :: l i o -> ()

data NFN i o = NFN

instance Category NFN where
  id = NFN
  NFN . NFN = NFN

instance (forall l. c l => NormalFormNetwork l) => NormalFormNetwork (Network c) where
  normalFormNetwork net = case free (\l -> normalFormNetwork l `seq` NFN) net of
    NFN -> ()

data FullyConnected i o where
  FullyConnected :: (KnownNat i, KnownNat o) => { weights :: V o (V i Double) } -> FullyConnected i o

instance FeedForwardCPU FullyConnected where
  feedForwardCPU (FullyConnected {weights}) v = weights !* v

instance BackPropagateCPU FullyConnected where
  backPropagateCPU learningRate FullyConnected {weights} (ia, og) =
    let ig = transpose weights !* og
    in ((weights !* ia, ig), FullyConnected {weights = weights !+! learningRate *!! og `outer` ia})

instance RandomInitialize FullyConnected where
  randomInitialize FullyConnected {weights} gen = FullyConnected <$> traverse (traverse (\_ -> standard gen)) weights

instance NormalFormNetwork FullyConnected where
  normalFormNetwork FullyConnected {weights} = rnf weights

data AddBiases i o where
  AddBiases :: KnownNat x => { biases :: V x Double } -> AddBiases x x

instance FeedForwardCPU AddBiases where
  feedForwardCPU AddBiases {biases} v = v ^+^ biases

instance BackPropagateCPU AddBiases where
  backPropagateCPU learningRate AddBiases {biases} (ia, og) =
    ((ia ^+^ biases, og), AddBiases {biases = biases + learningRate *^ og})

instance RandomInitialize AddBiases where
  randomInitialize AddBiases {biases} gen = AddBiases <$> traverse (\_ -> standard gen) biases

instance NormalFormNetwork AddBiases where
  normalFormNetwork AddBiases {biases} = rnf biases

data Sigmoid i o where
  Sigmoid :: KnownNat x => Sigmoid x x

sigmoidFunction :: Double -> Double
sigmoidFunction x = 1/(1 + exp (negate x))

instance FeedForwardCPU Sigmoid where
  feedForwardCPU Sigmoid = fmap sigmoidFunction

instance BackPropagateCPU Sigmoid where
  backPropagateCPU _ Sigmoid (ia, og) =
    let oa = fmap sigmoidFunction ia
        ig = fmap (\x -> sigmoidFunction x * (1 - sigmoidFunction x)) ia * og
    in ((oa, ig), Sigmoid)

instance RandomInitialize Sigmoid where
  randomInitialize Sigmoid _ = pure Sigmoid

instance NormalFormNetwork Sigmoid where
  normalFormNetwork Sigmoid = ()

-- training
--
-- List of (input, output) pairs
-- final cost function: quadratic difference
-- hooking it all together


train 
  :: (forall l. c l => (BackPropagateCPU l, NormalFormNetwork l)
     ,KnownNat o
     )
  => [(V i Double, V o Double)]
  -> Double
  -> Network c i o
  -> Network c i o
train dat learningRate n = foldl' trainExample' n dat
  where
    trainExample net (input, ref) = 
      let ((output, _), net') = backPropagateCPU learningRate net (input, cost)
          cost = output - ref
      in net'

    trainExample' net ir =
      let net' = trainExample net ir
      in normalFormNetwork net' `seq` net'

class (BackPropagateCPU l, NormalFormNetwork l, RandomInitialize l) => MNistC l
instance (BackPropagateCPU l, NormalFormNetwork l, RandomInitialize l) => MNistC l

mnist :: Network MNistC (28 GHC.TypeNats.* 28) 10
mnist =
  FullyConnected (pure (pure 0) :: V 15 (V 784 Double)) `Cons`
  AddBiases (pure 0) `Cons`
  Sigmoid `Cons`
  FullyConnected (pure (pure 0)) `Cons`
  AddBiases (pure 0) `Cons`
  Sigmoid `Cons`
  Nil
