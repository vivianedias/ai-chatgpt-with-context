// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import type { NextApiRequest, NextApiResponse } from "next";

import {
  IndexDict,
  RetrieverQueryEngine,
  TextNode,
  VectorStoreIndex,
} from "llamaindex";
import nodes from "../../public/storage/nodes";

type Input = {
  query: string;
};

type Output = {
  error?: string;
  payload?: {
    response: string;
  };
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<Output>
) {
  try {
    if (req.method !== "POST") {
      res.status(405).json({ error: "Method not allowed" });
      return;
    }

    const { query }: Input = req.body;
    const embeddingResults = nodes.map((config) => {
      return {
        node: new TextNode({ text: config.text }),
        embedding: config.embedding,
      };
    });
    const indexDict = new IndexDict();
    for (const { node } of embeddingResults) {
      indexDict.addNode(node);
    }

    const index = await VectorStoreIndex.init({ indexStruct: indexDict });

    index.vectorStore.add(embeddingResults);
    if (!index.vectorStore.storesText) {
      await index.docStore.addDocuments(
        embeddingResults.map((result) => result.node),
        true
      );
    }
    await index.indexStore?.addIndexStruct(indexDict);
    index.indexStruct = indexDict;

    const retriever = index.asRetriever();
    retriever.similarityTopK = 2;
    const queryEngine = new RetrieverQueryEngine(retriever);
    const result = await queryEngine.query(query);

    res.status(200).json({ payload: { response: result.response } });
  } catch (e) {
    console.log(e);
    return res.status(400).json({
      payload: {
        response: "Houve um erro",
      },
    });
  }
}
