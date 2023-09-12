// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import type { NextApiRequest, NextApiResponse } from "next";
import {
  MetadataMode,
  PDFReader,
  SentenceSplitter,
  VectorStoreIndex,
  getNodesFromDocument,
  serviceContextFromDefaults,
} from "llamaindex";

// type Input = {
//   document: string;
//   chunkSize?: number;
//   chunkOverlap?: number;
// };

type Output = {
  error?: string;
  payload?: {
    nodesWithEmbedding: {
      text: string;
      embedding: number[];
    }[];
  };
};

const DEFAULT_CHUNK_SIZE = 1024;
const DEFAULT_CHUNK_OVERLAP = 20;
// const DEFAULT_TOP_K = 2;

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<Output>
) {
  // const { document, chunkSize, chunkOverlap }: Input = req.body;

  const reader = new PDFReader();
  const document1 = await reader.loadData("public/assets/protocolo-1.pdf");
  const document2 = await reader.loadData("public/assets/protocolo-2.pdf");
  const document3 = await reader.loadData("public/assets/protocolo-3.pdf");

  const nodes = [...document1, ...document2, ...document3]
    .map((document) => {
      return getNodesFromDocument(
        document,
        new SentenceSplitter(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
      );
    })
    .flat();

  const nodesWithEmbeddings = await VectorStoreIndex.getNodeEmbeddingResults(
    nodes,
    serviceContextFromDefaults(),
    true
  );

  const nodesWithEmbedding = nodesWithEmbeddings.map((nodeWithEmbedding) => ({
    text: nodeWithEmbedding.node.getContent(MetadataMode.NONE),
    embedding: nodeWithEmbedding.embedding,
  }));

  res.status(200).json({
    payload: {
      nodesWithEmbedding,
    },
  });
}
