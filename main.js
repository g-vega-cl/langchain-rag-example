import "cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { pull } from "langchain/hub";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { formatDocumentsAsString } from "langchain/util/document";
import { RunnableSequence, RunnablePassthrough, } from "@langchain/core/runnables";
import { GithubRepoLoader } from "@langchain/community/document_loaders/web/github";
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
// ES module equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoUrl = "https://github.com/g-vega-cl/u-n-a";
const repoBranch = "main";
const docsFileName = `docs-${encodeURIComponent(repoUrl)}-${repoBranch}.json`;
const loadDocsFromFile = (filePath) => {
    if (fs.existsSync(filePath)) {
        const data = fs.readFileSync(filePath, 'utf-8');
        return JSON.parse(data);
    }
    return null;
};
const saveDocsToFile = (filePath, docs) => {
    fs.writeFileSync(filePath, JSON.stringify(docs, null, 2));
};
const filePath = path.join(__dirname, docsFileName);
let docs = loadDocsFromFile(filePath);
if (!docs) {
    console.log('loading docs from loader');
    const loader = new GithubRepoLoader(repoUrl, {
        branch: repoBranch,
        recursive: true,
        unknown: "warn",
        maxConcurrency: 5, // Defaults to 2
    });
    docs = await loader.load();
    saveDocsToFile(filePath, docs);
    console.log('docs', docs);
}
else {
    console.log('Docs loaded from file:', docsFileName);
}
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});
const splits = await textSplitter.splitDocuments(docs);
const vectorStore = await MemoryVectorStore.fromDocuments(splits, new OpenAIEmbeddings());
// Retrieve and generate using the relevant snippets of the blog.
const retriever = vectorStore.asRetriever();
const prompt = await pull("rlm/rag-prompt");
const llm = new ChatOpenAI({ model: "gpt-4-turbo", temperature: 0.5 });
const declarativeRagChain = RunnableSequence.from([
    {
        context: retriever.pipe(formatDocumentsAsString),
        question: new RunnablePassthrough(),
    },
    prompt,
    llm,
    new StringOutputParser(),
]);
const query = `Look at CreditCardSelector.tsx, PaymentFormButton.tsx, PurchasePlanCheckoutScreen.tsx, PlanDetailsScreen.tsx, and any files related. 
let me know if the states of the files conflict with each other, or if there is a clear bug when clicking a button. 
Assume that network calls can fail and those errors need to be handled.

Give me the best practices and refactorings I can make to those files and related files.
`;
const answer = await declarativeRagChain.invoke(query);
console.log('----------------------------------------------------------');
console.log(query);
console.log('');
console.log(answer);
