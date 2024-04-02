const express = require("express");
const cors = require("cors");
const axios = require("axios");
const { ReadFileTool, SerpAPI } = require("langchain/tools");
require("dotenv").config();
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");

const bodyParser = require("body-parser");
const { ChatOpenAI } = require("langchain/chat_models/openai");
const { PineconeClient } = require("@pinecone-database/pinecone");
const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { OpenAI } = require("langchain/llms/openai");
const PORT = process.env.PORT || 3005;
const app = express();

app.use(cors());

app.use(
  bodyParser.json({
    limit: "50mb",
  })
);
app.use(
  bodyParser.urlencoded({
    limit: "50mb",
    parameterLimit: 100000,
    extended: true,
  })
);
const model = new OpenAI({
  openAIApiKey: process.env.API_KEY3,
  temperature: 0.5,
  modelName: "gpt-3.5-turbo",
  streaming: true,
  cache: true,
});

async function initPinecone() {
  try {
    const pinecone = await new PineconeClient();

    await pinecone.init({
      environment: process.env.PINECONE_ENV, //this is in the dashboard
      apiKey: process.env.PINECONE_API_KEY,
    });

    return pinecone;
  } catch (error) {
    console.log("error", error);
    throw new Error(
      "Failed to initialize Pinecone Client, please make sure you have the correct environment and api keys"
    );
  }
}
// app.get("/try", async (req, res) => {
//   const embeddingsArrays = await new OpenAIEmbeddings({
//     openAIApiKey: process.env.OPEN_API_EMBEDDING,
//   }).embedDocuments([
//     `A flower is the reproductive part of a flowering plant. Flowers are also known as the bloom or blossom of a plant.
//     Flowers have petals, and inside the part of the flower that has petals are the parts which produce pollen and seeds. Flowers produce gametophytes, which in flowering plants consist of a few haploid cells which produce gametes.
//     Flowers are often brightly colored, grow at the end of a stem, and only survive for a short time. Each individual flower is tiny.
//     `,
//   ]);
//   console.log(embeddingsArrays);
//   res.json(embeddingsArrays);
// });

//set the data in pinecone
app.post("/set_file_to_pinecone", async (req, res) => {
  const { email, combinedContent } = req.body;
  if (email) {
    try {
      const embeddingsArrays = await new OpenAIEmbeddings({
        openAIApiKey: process.env.OPEN_API_EMBEDDING,
      }).embedDocuments(
        combinedContent.map((chunk) => chunk.pageContent.replace(/\n/g, " "))
      );
      console.log("embeddingsArrays", embeddingsArrays);
      const client = await initPinecone();
      const pinecone_indx = await client.Index("chatbot");
      const batchSize = 100;
      var batch = [];
      for (var idx = 0; idx < combinedContent.length; idx++) {
        const chunk = combinedContent[idx];
        const vector = {
          id: `${combinedContent[idx]["metadata"]["source"]}_${idx}`,
          values: embeddingsArrays[idx],
          metadata: {
            ...chunk.metadata,
            loc: JSON.stringify(chunk.metadata.loc),
            pageContent: chunk.pageContent,
            txtPath: combinedContent[idx]["metadata"]["source"],
          },
        };
        batch = [...batch, vector];
        if (batch.length === batchSize || idx === combinedContent.length - 1) {
          const result = await pinecone_indx.upsert({
            upsertRequest: {
              vectors: batch,
              // namespace: email,
            },
          });
          // Empty the batch
          console.log("result", result);
          batch = [];
        }
      }

      res.send({ message: "Code stored successfully" });
    } catch (error) {
      console.log("error", error);
      res.status(500).send({ message: "Error in storing code" });
    }
  }
  //no use
});

//this will be for clear history
app.delete("/deletepinecone", async (req, res) => {
  const { email } = req.body;
  if (email) {
    console.log("deletepinecone", email);
    try {
      const client = await initPinecone();
      const pinecone_indx = await client.Index("chatbot");
      await pinecone_indx._delete({
        deleteRequest: {
          deleteAll: true,
          // namespace: email,
        },
      });
      res.send({ message: "Code deleted successfully" });
    } catch (error) {
      console.log("error", error);
      res.status(500).send({ message: "Error in deleting code" });
    }
  } else {
    res.status(400).send({ message: "Email not found" });
  }
});

app.post("/image-text-chat", async (req, res) => {
  var data = req.body.data;
  var type = req.body.type;
  data = JSON.parse(data);
  console.log(data[0].generated_text, "data");
  var prompt = `You are a experienced Chat assistant, Explain in short \n Reference: ${data[0].generated_text} as accurately as possible, the given reference is a ${type} data, understand it properly and give the context about it`;
  await model.call(prompt, undefined, [
    {
      handleLLMNewToken(token) {
        console.log(token);
        res.write(token);
      },
    },
  ]);

  res.end();
});
app.post("/video-text-chat", async (req, res) => {
  console.log("video-text-chat");
  var title = req.body.title;
  var channel_name = req.body.channel_name;

  var prompt = `Give the summary of this video ${title} which is a title from youtube video understand by yourself and provide the context in 4 lines,This is the ${channel_name} name of this video. Don'nt ask user for more information just provide the summary`;
  await model.call(prompt, undefined, [
    {
      handleLLMNewToken(token) {
        console.log(token);
        res.write(token);
      },
    },
  ]);

  res.end();
});
app.post("/chat_with_ai", async (req, res) => {
  var { message, email } = req.body;
  console.log(req.body);
  console.log("semantics search");
  const client = await initPinecone();
  const index = await client.Index("chatbot");
  // if (email) {
  const queryEmbedding = await new OpenAIEmbeddings({
    openAIApiKey: process.env.API_KEY3,
  }).embedQuery(message);
  let queryResponse = await index.query({
    queryRequest: {
      topK: 1,
      vector: queryEmbedding,
      includeMetadata: true,
      includeValues: true,
    },
  });
  var concatenatedPageContent;
  if (queryResponse.matches.length) {
    concatenatedPageContent = queryResponse.matches[0].metadata.pageContent;
    //  Extract and concatenate page content from matched documents
    concatenatedPageContent = queryResponse.matches
      .map((match) => match.metadata.pageContent)
      .join(" ");
  }
  console.log("semantics search", concatenatedPageContent);
  // console.log("length", queryResponse.matches.length);

  var prompt = `You are a experienced Chat assistant, Answer the given query \n ${message} as accurately as possible`;

  if (concatenatedPageContent) {
    prompt = `You are a experienced Chat assistant, Answer the given query \n ${message}\n as accurately as possible, You can use the reference below \n${concatenatedPageContent}\n to find the related in the user's document if required so that you can be useful! Only use this reference if it is related to the given query, else ignore it and only answer the given query as accurately as possible. DO NOT ask the user to "locate" or "provide the content of" a file. find it yourself and read it using the ${concatenatedPageContent}\n. Read multiple times if necessary. the point is for YOU to do the work, NOT the user. try to find and read files as much as you need to. figure out the solution yourself. Locate all relevant data (and read them) YOURSELF please, don't ask the user to provide the content of a file because they can't. Also make sure to actually help the user solve their problem by going deep dive into its query and finding out the best possible solution for them like an experienced assistant. You will be operating in many different directories so the path to the file you need to read may be different each time. `;
    if (prompt && prompt.length > 8192) {
      prompt = prompt.substr(0, 8192);
    }
  }

  await model.call(prompt, undefined, [
    {
      handleLLMNewToken(token) {
        console.log(token);
        res.write(token);
      },
    },
  ]);

  res.end();
  // }
});
app.listen(PORT, () => console.log(`Listening on port ${PORT}`));
