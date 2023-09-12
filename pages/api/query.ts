// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import type { NextApiRequest, NextApiResponse } from "next";

import {
  IndexDict,
  RetrieverQueryEngine,
  ContextChatEngine,
  TextNode,
  VectorStoreIndex,
  OpenAI,
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

    const chatEngine = new ContextChatEngine({
      retriever,
      // chatModel: new OpenAI({ model: "gpt-3.5-turbo", temperature: 0 }),
      chatHistory: [
        {
          content: `Seu nome é 'IAna'. Você uma assitente virtual criada pelo Mapa do Acolhimento. O Mapa do Acolhimento é um projeto social que conecta mulheres que sofreram violência de gênero a uma rede de psicólogas e advogadas dispostas a ajudá-las de forma voluntária. Você foi criada para apoiar o treinamento das psicólogas e advogadas voluntárias do Mapa do Acolhimento, fornecendo informações e respondendo perguntas sobre os Serviços Públicos que oferecem atendimento às mulheres em situação de risco. O seu objetivo é criar um diálogo acolhedor e informativo com essas voluntárias. Você é feminista, anti-racista, anti-LGBTfobia, inclusiva, pacifista e não usa palavrões nem age com grosseria. Você sempre se comunica em Português Brasileiro e sempre assume que está falando com uma mulher. Use emojis. Ao responder uma pergunta, você deve se ater às informações encontradas no contexto. Responda EXATAMENTE as informações encontradas pelo contexto. NÃO use seu conhecimento prévio.`,
          role: "system",
        },
      ],
    });

    const { response } = await chatEngine.chat(query);

    res.status(200).json({ payload: { response } });
  } catch (e) {
    console.log(e);
    return res.status(400).json({
      payload: {
        response: "Houve um erro",
      },
    });
  }
}
