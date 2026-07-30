# RAG Chatbot

Upload documents, then ask questions grounded in exactly what's in them.
Built phase by phase to actually learn how RAG (Retrieval-Augmented
Generation) works, not just wire an existing library together.
POST /documents/upload Upload a .pdf/.docx/.txt/.md file
GET /documents List uploaded documents
GET /documents/{doc_id} Full detail for one document, including its chunks
DELETE /documents/{doc_id} Remove a document (vectors AND the file on disk)
POST /chat Ask a question, get a grounded answer back
POST /chat/stream Same, but streamed token-by-token