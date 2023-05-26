// 1. Import necessary modules and libraries
import { OpenAI } from 'langchain/llms';
import { RetrievalQAChain } from 'langchain/chains';
import { HNSWLib } from 'langchain/vectorstores';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from 'fs';
import * as dotenv from 'dotenv';

// 2. Load environment variables
dotenv.config();

// 3. Set up input data and paths
const txtFilename = "pokemon";
const question = `
For database schema as '[{\"id\":10000,\"hint\":\"\",\"name\":\"Name\",\"type\":\"Text\"},{\"id\":\"10001\",\"hint\":\"\",\"name\":\"Type 1\",\"type\":\"Text\"},{\"id\":\"10002\",\"hint\":\"\",\"name\":\"Type 2\",\"type\":\"Text\"},{\"id\":\"10004\",\"hint\":\"\",\"name\":\"Total\",\"type\":\"Digit\"},{\"id\":\"10005\",\"hint\":\"\",\"name\":\"HP\",\"type\":\"Digit\"},{\"id\":\"10006\",\"hint\":\"\",\"name\":\"Attack\",\"type\":\"Digit\"},{\"id\":\"10007\",\"hint\":\"\",\"name\":\"Defense\",\"type\":\"Digit\"},{\"id\":\"10009\",\"hint\":\"\",\"name\":\"Sp. Atk\",\"type\":\"Digit\"},{\"id\":\"10010\",\"hint\":\"\",\"name\":\"Sp. Def\",\"type\":\"Digit\"},{\"id\":\"10011\",\"hint\":\"\",\"name\":\"Generation\",\"type\":\"Digit\"},{\"id\":\"10012\",\"hint\":\"\",\"name\":\"Legendary\",\"type\":\"Single Select\",\"config\":{\"enableOtherOption\":false},\"options\":[\"True\",\"False\"]}]', translate the query into a sql, and reply as follows:
{"sql": "sql"}
Example:
{"sql": "select count(*) from pokemon where generation=1;"}

If you do not know the answer, reply as follows:
{"answer": "I do not know."}

Return all output as a string.

Lets think step by step.

Below is the query.
Query: 
` + "how many pokemon are generation 2?";
// const question = `
// For the following query, if it requires drawing a table, reply as follows:
// {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

// If the query requires creating a bar chart, reply as follows:
// {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

// If the query requires creating a line chart, reply as follows:
// {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

// There can only be two types of chart, "bar" and "line".

// If it is just asking a question that requires neither, reply as follows:
// {"answer": "answer"}
// Example:
// {"answer": "The title with the highest rating is 'Gilead'"}

// If you do not know the answer, reply as follows:
// {"answer": "I do not know."}

// Return all output as a string.

// All strings in "columns" list and data list, should be in double quotes,

// For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

// Lets think step by step.

// Below is the query.
// Query: 
// ` + "how many pokemon are generation 1?";
const txtPath = `./${txtFilename}.csv`;
const VECTOR_STORE_PATH = `${txtFilename}.index`;

// 4. Define the main function runWithEmbeddings
export const runWithEmbeddings = async () => {
  // 5. Initialize the OpenAI model with an empty configuration object
  const model = new OpenAI({});

  // 6. Check if the vector store file exists
  let vectorStore;
  if (fs.existsSync(VECTOR_STORE_PATH)) {
    // 6.1. If the vector store file exists, load it into memory
    console.log('Vector Exists..');
    vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
  } else {
    // 6.2. If the vector store file doesn't exist, create it
    // 6.2.1. Read the input text file
    const text = fs.readFileSync(txtPath, 'utf8');
    // 6.2.2. Create a RecursiveCharacterTextSplitter with a specified chunk size
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
    // 6.2.3. Split the input text into documents
    const docs = await textSplitter.createDocuments([text]);
    // 6.2.4. Create a new vector store from the documents using OpenAIEmbeddings
    vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
    // 6.2.5. Save the vector store to a file
    await vectorStore.save(VECTOR_STORE_PATH);
  }

  // 7. Create a RetrievalQAChain by passing the initialized OpenAI model and the vector store retriever
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

  // 8. Call the RetrievalQAChain with the input question, and store the result in the 'res' variable
  const res = await chain.call({
    query: question,
  });

  // 9. Log the result to the console
  console.log({ res });
};

// 10. Execute the main function runWithEmbeddings
runWithEmbeddings();
