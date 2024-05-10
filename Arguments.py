import argparse
import os


def argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--proteinFilterLen',
      type=int,
      default=11,
      help='The filter lenth of convolution for decoder protein'
  )
  parser.add_argument(
      '--drugFilterLen',
      type=int,
      default=7,
      help='The filter lenth of convolution for decoder drug'
  )
  parser.add_argument(
      '--numHidden',
      type=int,
      default=32,
      help='The hidden nums of model'
  )

  parser.add_argument(
      '--maxProteinLen',
      type=int,
      default=1200,
      help='Length of input protein.'
  )
  parser.add_argument(
      '--maxDrugLen',
      type=int,
      default=85,
      help='Length of input drug.'
  )

  parser.add_argument(
      '--numEpoch',
      type=int,
      default=1,
      help='Number of epochs to train.'
  )
  parser.add_argument(
      '--batchSize',
      type=int,
      default=256,
      help='Batch size.'
  )
  parser.add_argument(
      '--datasetPath',
      type=str,
      default='data/davis/',
      help='Directory for input data.'
  )
  parser.add_argument(
      '--problemType',
      type=int,
      default=2,
      help='Type of the prediction problem (1-3)'
  )

  parser.add_argument(
      '--logEnable',
      type=int,
      default=1,
      help='Use log to record info'
  )
  parser.add_argument(
      '--checkpointPath',
      type=str,
      default='',
      help='Path to write checkpoint file.'
  )
  parser.add_argument(
      '--logDir',
      type=str,
      default='NewLogFile/',
      help='Directory for log data.'
  )
  parser.add_argument(
      '--lamda',
      type=int,
      default=-5,
  )
  parser.add_argument(
        '--GPU',
        type=int,
        default=2,
  )
  parser.add_argument(
        '--modelName',
        type=str,
        default="TIVAE",
        help = "CoVAE, TIVAE, MTDTA"
  )
  parser.add_argument(
        '--gitNode',
        type=str,
        default="NULL"
  )
  parser.add_argument(
        '--MoreInfo',
        type=str,
        default="TIVAE-3CNN->1GCN-I",
  )
  FLAGS, unparsed = parser.parse_known_args()
  return FLAGS
